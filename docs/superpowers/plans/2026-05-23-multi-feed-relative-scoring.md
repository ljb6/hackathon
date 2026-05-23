# Multi-Feed + Relative Scoring — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support multiple simultaneous camera feeds returning per-camera best matches, and fix false positives by replacing absolute cosine similarity with relative scoring (query delta vs a generic "a person" baseline).

**Architecture:** `search.py` gains an optional `baseline_embedding` parameter on both `find_best_match` and `is_match`. When provided, the score becomes `cosine_sim(crop, query) - cosine_sim(crop, baseline)` — a delta that is near zero when CLIP can't distinguish the query from a generic person, and positive when the crop genuinely matches. `main.py` computes `baseline_emb = embed_text("a person")` once at startup, adds `_collect_candidates(source, ...)` to separate processing from display, and adds `run_multi_video` to loop over a list of paths and collect per-camera results.

**Tech Stack:** Same as clip-person-search — torch, CLIP, YOLOv8, opencv-python

---

## File Map

| File | Change |
|------|--------|
| `clip-person-search/search.py` | Add `baseline_embedding` param to `find_best_match` + `is_match`; add `RELATIVE_THRESHOLD = 0.02` |
| `clip-person-search/main.py` | Add `_collect_candidates`, `run_multi_video`; update `run_video`, `run_live`, `main` to pass `baseline_emb` |
| `clip-person-search/tests/test_search.py` | Add 4 tests for relative scoring |
| `clip-person-search/tests/test_main.py` | No new tests needed (run_multi_video requires VideoCapture — smoke-tested in Task 3) |

---

## Task 1: search.py — Relative Scoring

### Why absolute scoring fails

CLIP assigns similar absolute cosine scores (~0.20-0.23) to any person regardless of whether the description matches. The relative score `cosine_sim(crop, "purple shirt") - cosine_sim(crop, "a person")` is near zero when nobody matches, and rises to ~0.03-0.08 only when the person actually matches the description.

**Files:**
- Modify: `clip-person-search/search.py`
- Modify: `clip-person-search/tests/test_search.py`

- [ ] **Step 1: Write the new failing tests**

Add these tests to `clip-person-search/tests/test_search.py` (append after existing tests):

```python
# --- relative scoring (baseline_embedding provided) ---

def test_find_best_match_relative_no_match_when_delta_below_threshold():
    # When query and baseline are identical, delta = 0 → no match
    emb_query    = torch.tensor([[1.0, 0.0]])
    emb_baseline = torch.tensor([[1.0, 0.0]])  # same direction as query
    emb_img      = torch.tensor([[1.0, 0.0]])  # matches both equally

    candidates = [_candidate(emb_img)]
    result = find_best_match(emb_query, candidates, threshold=0.01,
                             baseline_embedding=emb_baseline)
    assert result is None  # delta = 1.0 - 1.0 = 0.0 < 0.01


def test_find_best_match_relative_returns_match_when_delta_positive():
    # Query matches crop; baseline is orthogonal → delta ≈ 1.0
    emb_query    = torch.tensor([[1.0, 0.0]])
    emb_baseline = torch.tensor([[0.0, 1.0]])  # orthogonal
    emb_img      = torch.tensor([[1.0, 0.0]])  # matches query perfectly

    candidates = [_candidate(emb_img)]
    result = find_best_match(emb_query, candidates, threshold=0.01,
                             baseline_embedding=emb_baseline)
    assert result is not None
    _, _, score, _ = result
    assert score > 0.9  # delta ≈ 1.0 - 0.0


def test_is_match_relative_no_match_when_equal_scores():
    emb_query    = torch.tensor([[1.0, 0.0]])
    emb_baseline = torch.tensor([[1.0, 0.0]])
    emb_img      = torch.tensor([[1.0, 0.0]])

    score, matched = is_match(emb_query, emb_img, threshold=0.01,
                              baseline_embedding=emb_baseline)
    assert matched is False
    assert abs(score) < 0.001  # delta ≈ 0


def test_is_match_relative_match_when_query_more_similar():
    emb_query    = torch.tensor([[1.0, 0.0]])
    emb_baseline = torch.tensor([[0.0, 1.0]])
    emb_img      = torch.tensor([[1.0, 0.0]])

    score, matched = is_match(emb_query, emb_img, threshold=0.01,
                              baseline_embedding=emb_baseline)
    assert matched is True
    assert score > 0.9
```

- [ ] **Step 2: Run tests to confirm the new ones fail**

```bash
cd /Users/lucacarvalhojeo/development/Hackathon_Seg_Publica/hackathon/clip-person-search && python -m pytest tests/test_search.py -v -k "relative"
```

Expected: `4 failed` (baseline_embedding not yet accepted)

- [ ] **Step 3: Update search.py**

Replace the entire contents of `clip-person-search/search.py`:

```python
import torch.nn.functional as F

SIMILARITY_THRESHOLD = 0.20   # legacy absolute mode
RELATIVE_THRESHOLD   = 0.02   # delta mode: cosine_sim(crop, query) - cosine_sim(crop, baseline)


def find_best_match(query_embedding, candidates, threshold=RELATIVE_THRESHOLD,
                    baseline_embedding=None):
    """
    Batch mode. candidates: list of (crop, box, frame, image_embedding, timestamp).
    Returns (frame, box, score, timestamp) for the best candidate above threshold,
    or None if no candidate clears it.

    When baseline_embedding is provided, score = cosine_sim(img, query) - cosine_sim(img, baseline).
    This relative delta is near zero when CLIP cannot distinguish the query from a generic
    person, and positive when the crop genuinely matches the description.
    """
    if not candidates:
        return None

    best_score = -float("inf")
    best = None

    for crop, box, frame, img_emb, timestamp in candidates:
        specific = F.cosine_similarity(query_embedding, img_emb).item()
        if baseline_embedding is not None:
            generic = F.cosine_similarity(baseline_embedding, img_emb).item()
            score = specific - generic
        else:
            score = specific

        if score > best_score:
            best_score = score
            best = (frame, box, score, timestamp)

    return best if best_score >= threshold else None


def is_match(query_embedding, img_embedding, threshold=RELATIVE_THRESHOLD,
             baseline_embedding=None):
    """
    Live mode. Returns (score: float, matched: bool).
    When baseline_embedding is provided, score is the relative delta.
    """
    specific = F.cosine_similarity(query_embedding, img_embedding).item()
    if baseline_embedding is not None:
        generic = F.cosine_similarity(baseline_embedding, img_embedding).item()
        score = specific - generic
    else:
        score = specific
    return score, score >= threshold
```

- [ ] **Step 4: Run full search test suite**

```bash
cd /Users/lucacarvalhojeo/development/Hackathon_Seg_Publica/hackathon/clip-person-search && python -m pytest tests/test_search.py -v
```

Expected: `11 passed` (7 existing + 4 new)

- [ ] **Step 5: Commit**

```bash
git -C /Users/lucacarvalhojeo/development/Hackathon_Seg_Publica/hackathon add clip-person-search/search.py clip-person-search/tests/test_search.py
git -C /Users/lucacarvalhojeo/development/Hackathon_Seg_Publica/hackathon commit -m "feat: add relative scoring to eliminate false positives"
```

---

## Task 2: main.py — Multi-Feed + Baseline Threading

New functions:
- `_collect_candidates(source, query_emb, baseline_emb) -> (camera_label, result | None)` — separates processing from display; returns per-camera best result
- `run_multi_video(sources, query_emb, query, baseline_emb)` — loops sources, collects results, displays all matches

Updated functions:
- `run_video(source, query_emb, query, baseline_emb)` — now delegates to `_collect_candidates`, passes `baseline_emb` to `find_best_match`
- `run_live(query_emb, query, baseline_emb)` — passes `baseline_emb` to `is_match`
- `main()` — parses comma-separated sources, computes `baseline_emb` once, dispatches

**Files:**
- Modify: `clip-person-search/main.py`

- [ ] **Step 1: Verify existing main tests still pass before touching the file**

```bash
cd /Users/lucacarvalhojeo/development/Hackathon_Seg_Publica/hackathon/clip-person-search && python -m pytest tests/test_main.py -v
```

Expected: `7 passed`

- [ ] **Step 2: Replace main.py**

Replace the entire contents of `clip-person-search/main.py`:

```python
import os
import cv2
from datetime import datetime

from query_builder import prompt_query
from detector import extract_persons
from embedder import embed_image, embed_text
from search import find_best_match, is_match
from tracker import PersonTracker

SAMPLE_EVERY = 15  # process every N frames


def timestamp_from_frame(frame_count, fps):
    total_seconds = int(frame_count / fps)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def draw_result(frame, box, score, query, timestamp):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    label = f"MATCH {score:.3f} | {query} | {timestamp}"
    cv2.putText(frame, label, (x1, max(y1 - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


def _collect_candidates(source, query_emb, baseline_emb):
    """
    Process one video source. Returns (camera_label, result) where result is
    (frame, box, score, timestamp) or None if no match above threshold.
    camera_label is the filename stem (e.g. "passageway1-c1").
    """
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    camera_label = os.path.splitext(os.path.basename(source))[0]
    candidates = []
    frame_count = 0
    tracker = PersonTracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % SAMPLE_EVERY != 0:
            continue

        timestamp = timestamp_from_frame(frame_count, fps)
        for crop, box in extract_persons(frame):
            img_emb, re_embedded = tracker.update(box, crop, frame_count, embed_image)
            if re_embedded:
                candidates.append((crop, box, frame.copy(), img_emb, timestamp))

    cap.release()

    if not candidates:
        return camera_label, None

    result = find_best_match(query_emb, candidates, baseline_embedding=baseline_emb)
    return camera_label, result


def run_video(source, query_emb, query, baseline_emb):
    """Single camera batch mode."""
    print(f"Processing {source}...")
    camera_label, result = _collect_candidates(source, query_emb, baseline_emb)

    if result is None:
        print(f"[{camera_label}] No match found above threshold.")
        return

    frame, box, score, timestamp = result
    output = draw_result(frame, box, score, query, timestamp)
    filename = f"result_{camera_label}.jpg"
    cv2.imwrite(filename, output)
    print(f"[{camera_label}] Match! Score: {score:.3f} at {timestamp} — saved to {filename}")
    cv2.imshow(f"{camera_label} — {score:.3f} @ {timestamp}", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_multi_video(sources, query_emb, query, baseline_emb):
    """Process multiple cameras, display all per-camera best matches at the end."""
    per_camera = []
    for source in sources:
        print(f"Processing {source}...")
        camera_label, result = _collect_candidates(source, query_emb, baseline_emb)
        per_camera.append((camera_label, result))
        status = f"Score: {result[2]:.3f} @ {result[3]}" if result else "no match"
        print(f"  [{camera_label}] {status}")

    matched = [(label, r) for label, r in per_camera if r is not None]

    if not matched:
        print("\nNo matches found in any camera above similarity threshold.")
        return

    print(f"\n{len(matched)}/{len(sources)} cameras matched:")
    for camera_label, (frame, box, score, timestamp) in matched:
        output = draw_result(frame, box, score, query, timestamp)
        filename = f"result_{camera_label}.jpg"
        cv2.imwrite(filename, output)
        print(f"  [{camera_label}] Score: {score:.3f} at {timestamp} → {filename}")
        cv2.imshow(f"{camera_label} — {score:.3f} @ {timestamp}", output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_live(query_emb, query, baseline_emb):
    cap = cv2.VideoCapture(0)
    frame_count = 0
    tracker = PersonTracker()

    print("Live feed active. Query stored. Press Q to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if frame_count % SAMPLE_EVERY == 0:
            timestamp = datetime.now().strftime("%H:%M:%S")
            for crop, box in extract_persons(frame):
                img_emb, re_embedded = tracker.update(box, crop, frame_count, embed_image)
                if re_embedded:
                    score, matched = is_match(query_emb, img_emb,
                                              baseline_embedding=baseline_emb)
                    if matched:
                        output = draw_result(frame.copy(), box, score, query, timestamp)
                        cv2.imshow("MATCH FOUND", output)
                        filename = f"match_{timestamp.replace(':', '')}.jpg"
                        cv2.imwrite(filename, output)
                        print(f"Match! Score: {score:.3f} at {timestamp} — saved to {filename}")

        cv2.imshow("Live Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    sources_input = input(
        "Video file paths (comma-separated, or Enter for live webcam): "
    ).strip()
    query = prompt_query()

    print("Loading query embeddings...")
    query_emb    = embed_text(query)
    baseline_emb = embed_text("a person")

    if sources_input:
        sources = [s.strip() for s in sources_input.split(",") if s.strip()]
        if len(sources) == 1:
            run_video(sources[0], query_emb, query, baseline_emb)
        else:
            run_multi_video(sources, query_emb, query, baseline_emb)
    else:
        run_live(query_emb, query, baseline_emb)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run main tests to confirm nothing broke**

```bash
cd /Users/lucacarvalhojeo/development/Hackathon_Seg_Publica/hackathon/clip-person-search && python -m pytest tests/test_main.py -v
```

Expected: `7 passed` (draw_result and timestamp_from_frame signatures unchanged)

- [ ] **Step 4: Commit**

```bash
git -C /Users/lucacarvalhojeo/development/Hackathon_Seg_Publica/hackathon add clip-person-search/main.py
git -C /Users/lucacarvalhojeo/development/Hackathon_Seg_Publica/hackathon commit -m "feat: add multi-feed support and thread baseline embedding through pipeline"
```

---

## Task 3: Full Test Suite + Smoke Run

- [ ] **Step 1: Run full suite**

```bash
cd /Users/lucacarvalhojeo/development/Hackathon_Seg_Publica/hackathon/clip-person-search && python -m pytest tests/ -v
```

Expected: `42 passed` (38 existing + 4 new relative-scoring tests)

- [ ] **Step 2: Smoke-test with real videos**

```bash
cd /Users/lucacarvalhojeo/development/Hackathon_Seg_Publica/hackathon/clip-person-search
python main.py
# Video file paths: ../videos/passageway1-c1.mp4,../videos/passageway1-c2.mp4,../videos/passageway1-c3.mp4
# Upper clothing color: red
# Has backpack? n   Has hat? n   Extra:
# → Processes all 3 cameras. result_passageway1-c1.jpg etc. saved for cameras that match.
# → If nobody matches the description, prints "No matches found in any camera."
```

- [ ] **Step 3: Final commit**

```bash
git -C /Users/lucacarvalhojeo/development/Hackathon_Seg_Publica/hackathon add -A
git -C /Users/lucacarvalhojeo/development/Hackathon_Seg_Publica/hackathon commit -m "feat: multi-feed + relative scoring — 42 tests passing"
```

---

## Threshold Tuning Reference

| Score type | What it means | Tune in |
|---|---|---|
| `RELATIVE_THRESHOLD = 0.02` | Minimum delta between specific and generic | `search.py` |
| `IOU_THRESHOLD = 0.4` | Same-person overlap to reuse embedding | `tracker.py` |
| `SAMPLE_EVERY = 15` | Frames skipped between samples | `main.py` |

If you still see false positives with relative scoring, raise `RELATIVE_THRESHOLD` to `0.03`.
If real matches are being missed, lower it to `0.01`.
