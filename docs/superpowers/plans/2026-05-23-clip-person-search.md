# CLIP Person Search — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI tool that detects people in a video file or live webcam feed and finds visual matches for a natural language query using CLIP cosine similarity, returning the frame, bounding box, score, and timestamp.

**Architecture:** Two modes share the same detection/embedding pipeline. **Video mode**: batch — collect candidates, find the single best match above a similarity threshold. **Live mode**: streaming — the query is stored at startup and checked against every new detection. Both modes use a `PersonTracker` to deduplicate embeddings across frames: when a person is detected again in the same region (IoU ≥ 0.4), the tracker computes the new crop's sharpness (Laplacian variance); it re-embeds via CLIP only if the new crop is sharper than the cached one, otherwise the cached embedding is reused. `is_match` / candidate append only happen on re-embed, so a person standing still triggers one CLIP call, and a person walking closer triggers another only when the image actually improves. The query itself is composed from structured prompts (upper/lower clothing color, backpack, hat, extra description).

**Tech Stack:** Python, `ultralytics` (YOLOv8n), `openai/clip` (ViT-B/32), `torch`, `opencv-python`, `Pillow`, `numpy`, `pytest`

---

## File Map

| File | Responsibility |
|------|----------------|
| `clip-person-search/requirements.txt` | Pinned dependencies |
| `clip-person-search/query_builder.py` | `compose_query` (pure fn) + `prompt_query` (interactive); zero ML dependencies |
| `clip-person-search/search.py` | `find_best_match` (batch + threshold) + `is_match` (per-person, live) |
| `clip-person-search/detector.py` | YOLOv8 person detection → `(crop, box)` pairs |
| `clip-person-search/embedder.py` | CLIP image + text embedding |
| `clip-person-search/tracker.py` | `sharpness()` + `PersonTracker` — IoU matching + sharpness-gated re-embedding |
| `clip-person-search/main.py` | `timestamp_from_frame`, `draw_result`, `run_video`, `run_live`, `main` |
| `clip-person-search/tests/test_query_builder.py` | Unit tests for `compose_query` + `prompt_query` |
| `clip-person-search/tests/test_search.py` | Unit tests for threshold logic + timestamp in results |
| `clip-person-search/tests/test_detector.py` | Unit tests for person extraction (mocked YOLO) |
| `clip-person-search/tests/test_embedder.py` | Unit tests for embedding functions (mocked CLIP) |
| `clip-person-search/tests/test_tracker.py` | Unit tests for sharpness metric + PersonTracker behaviour |
| `clip-person-search/tests/test_main.py` | Unit tests for `timestamp_from_frame` + `draw_result` |

**Candidate tuple throughout the codebase:** `(crop, box, frame, image_embedding, timestamp)`
- `crop`: `np.ndarray` BGR
- `box`: `(x1, y1, x2, y2)` ints
- `frame`: `np.ndarray` BGR full frame
- `image_embedding`: `torch.Tensor` shape `(1, 512)`
- `timestamp`: `str` `"HH:MM:SS"`

---

## Task 1: Project Scaffold + Dependencies

**Files:**
- Create: `clip-person-search/requirements.txt`
- Create: `clip-person-search/tests/__init__.py`

- [ ] **Step 1: Create project directory and requirements file**

```bash
mkdir -p clip-person-search/tests
touch clip-person-search/tests/__init__.py
```

Write `clip-person-search/requirements.txt`:

```
ultralytics>=8.0.0
git+https://github.com/openai/CLIP.git
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
pytest>=7.4.0
```

- [ ] **Step 2: Install dependencies**

```bash
cd clip-person-search
pip install -r requirements.txt
```

Expected: all packages install without errors. CLIP (~350 MB) and YOLOv8n (~6 MB) download on first model load at runtime, not here.

- [ ] **Step 3: Verify imports**

```bash
python -c "import torch; import clip; print('torch', torch.__version__); print('clip ok')"
```

Expected:
```
torch 2.x.x
clip ok
```

- [ ] **Step 4: Commit**

```bash
cd clip-person-search
git add requirements.txt tests/__init__.py
git commit -m "feat: scaffold clip-person-search project"
```

---

## Task 2: query_builder.py — Structured Query Composition

`query_builder.py` has no ML dependencies, so its tests run instantly. Keeping it separate from `main.py` lets us import and test it without triggering YOLO/CLIP model loading.

`compose_query` is a pure function. `prompt_query` wraps it with `input()` calls.

**Files:**
- Create: `clip-person-search/query_builder.py`
- Test: `clip-person-search/tests/test_query_builder.py`

- [ ] **Step 1: Write the failing tests**

Create `clip-person-search/tests/test_query_builder.py`:

```python
from unittest.mock import patch
from query_builder import compose_query, prompt_query


# --- compose_query ---

def test_all_fields_present():
    q = compose_query(upper_color="red", lower_color="black",
                      has_backpack=True, has_hat=False, extra="beard")
    assert "red" in q
    assert "black" in q
    assert "backpack" in q
    assert "beard" in q
    assert "hat" not in q


def test_only_upper_color():
    q = compose_query(upper_color="blue", lower_color="",
                      has_backpack=False, has_hat=False, extra="")
    assert "blue" in q
    assert "backpack" not in q


def test_only_backpack():
    q = compose_query(upper_color="", lower_color="",
                      has_backpack=True, has_hat=False, extra="")
    assert "backpack" in q


def test_only_hat():
    q = compose_query(upper_color="", lower_color="",
                      has_backpack=False, has_hat=True, extra="")
    assert "hat" in q


def test_all_empty_returns_nonempty_string():
    q = compose_query(upper_color="", lower_color="",
                      has_backpack=False, has_hat=False, extra="")
    assert isinstance(q, str) and len(q) > 0


def test_both_accessories():
    q = compose_query(upper_color="", lower_color="",
                      has_backpack=True, has_hat=True, extra="")
    assert "backpack" in q and "hat" in q


def test_extra_appended():
    q = compose_query(upper_color="", lower_color="",
                      has_backpack=False, has_hat=False, extra="tall man")
    assert "tall man" in q


def test_upper_and_lower_both_present():
    q = compose_query(upper_color="white", lower_color="blue",
                      has_backpack=False, has_hat=False, extra="")
    assert "white" in q and "blue" in q


# --- prompt_query ---

def test_prompt_query_returns_string_with_all_fields():
    with patch("builtins.input", side_effect=iter(["red", "black", "y", "y", "tall man"])):
        q = prompt_query()
    assert "red" in q and "black" in q
    assert "backpack" in q and "hat" in q
    assert "tall man" in q


def test_prompt_query_skips_empty_fields():
    with patch("builtins.input", side_effect=iter(["", "", "n", "n", ""])):
        q = prompt_query()
    assert "backpack" not in q and "hat" not in q


def test_prompt_query_prints_composed_query(capsys):
    with patch("builtins.input", side_effect=iter(["blue", "", "y", "n", ""])):
        prompt_query()
    assert "blue" in capsys.readouterr().out
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd clip-person-search
pytest tests/test_query_builder.py -v
```

Expected: `ImportError: No module named 'query_builder'`

- [ ] **Step 3: Implement query_builder.py**

Create `clip-person-search/query_builder.py`:

```python
def compose_query(upper_color="", lower_color="", has_backpack=False, has_hat=False, extra=""):
    """
    Assembles a natural language description for CLIP from structured fields.
    Returns at minimum "person".
    """
    parts = []
    if upper_color:
        parts.append(f"{upper_color} shirt")
    if lower_color:
        parts.append(f"{lower_color} pants")
    if has_backpack:
        parts.append("backpack")
    if has_hat:
        parts.append("hat")
    if extra:
        parts.append(extra)

    if not parts:
        return "person"
    return "person with " + " and ".join(parts)


def prompt_query():
    """Asks the user structured questions and returns the composed query string."""
    print("\n--- Describe the suspect ---")
    upper_color  = input("Upper clothing color (e.g. red, blue, black) [Enter to skip]: ").strip()
    lower_color  = input("Lower clothing color (e.g. black, blue) [Enter to skip]: ").strip()
    has_backpack = input("Has backpack? (y/N): ").strip().lower() == "y"
    has_hat      = input("Has hat? (y/N): ").strip().lower() == "y"
    extra        = input("Any other description (e.g. beard, tall) [Enter to skip]: ").strip()

    query = compose_query(upper_color, lower_color, has_backpack, has_hat, extra)
    print(f'\nSearching for: "{query}"\n')
    return query
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd clip-person-search
pytest tests/test_query_builder.py -v
```

Expected:
```
11 passed in 0.XXs
```

- [ ] **Step 5: Commit**

```bash
cd clip-person-search
git add query_builder.py tests/test_query_builder.py
git commit -m "feat: add structured query composition in query_builder.py"
```

---

## Task 3: search.py — Similarity Ranking with Threshold

**Files:**
- Create: `clip-person-search/search.py`
- Test: `clip-person-search/tests/test_search.py`

- [ ] **Step 1: Write the failing tests**

Create `clip-person-search/tests/test_search.py`:

```python
import torch
from search import find_best_match, is_match

THRESHOLD = 0.5


def _candidate(emb, timestamp="00:00:01", frame_label="frame"):
    return (None, (0, 0, 50, 50), frame_label, emb, timestamp)


def test_returns_best_match_above_threshold():
    emb_query = torch.tensor([[1.0, 0.0]])
    candidates = [
        _candidate(torch.tensor([[0.0, 1.0]]), "00:00:01", "bad"),
        _candidate(torch.tensor([[1.0, 0.0]]), "00:00:05", "good"),
    ]
    result = find_best_match(emb_query, candidates, threshold=THRESHOLD)
    assert result is not None
    frame, box, score, timestamp = result
    assert frame == "good"
    assert score > 0.99
    assert timestamp == "00:00:05"


def test_returns_none_when_best_score_below_threshold():
    emb_query = torch.tensor([[1.0, 0.0]])
    candidates = [_candidate(torch.tensor([[0.0, 1.0]]))]
    assert find_best_match(emb_query, candidates, threshold=THRESHOLD) is None


def test_returns_none_for_empty_candidates():
    assert find_best_match(torch.tensor([[1.0, 0.0]]), [], threshold=THRESHOLD) is None


def test_result_carries_correct_timestamp():
    emb_query = torch.tensor([[1.0, 0.0]])
    candidates = [_candidate(torch.tensor([[1.0, 0.0]]), timestamp="00:02:47")]
    _, _, _, timestamp = find_best_match(emb_query, candidates, threshold=THRESHOLD)
    assert timestamp == "00:02:47"


def test_is_match_true_above_threshold():
    score, matched = is_match(
        torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 0.0]]), threshold=THRESHOLD)
    assert matched is True and score > 0.99


def test_is_match_false_below_threshold():
    score, matched = is_match(
        torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]]), threshold=THRESHOLD)
    assert matched is False and score < 0.01


def test_is_match_returns_float_score():
    score, _ = is_match(
        torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 0.0]]), threshold=THRESHOLD)
    assert isinstance(score, float)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd clip-person-search
pytest tests/test_search.py -v
```

Expected: `ImportError: No module named 'search'`

- [ ] **Step 3: Implement search.py**

Create `clip-person-search/search.py`:

```python
import torch.nn.functional as F

# CLIP text-image cosine similarity: ~0.10-0.20 for random pairs, 0.20-0.35 for matches.
SIMILARITY_THRESHOLD = 0.20


def find_best_match(query_embedding, candidates, threshold=SIMILARITY_THRESHOLD):
    """
    Batch mode. candidates: list of (crop, box, frame, image_embedding, timestamp).
    Returns (frame, box, score, timestamp) for the best candidate above threshold,
    or None if no candidate clears it.
    """
    if not candidates:
        return None

    best_score = -float("inf")
    best = None

    for crop, box, frame, img_emb, timestamp in candidates:
        score = F.cosine_similarity(query_embedding, img_emb).item()
        if score > best_score:
            best_score = score
            best = (frame, box, score, timestamp)

    return best if best_score >= threshold else None


def is_match(query_embedding, img_embedding, threshold=SIMILARITY_THRESHOLD):
    """Live mode. Returns (score: float, matched: bool)."""
    score = F.cosine_similarity(query_embedding, img_embedding).item()
    return score, score >= threshold
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd clip-person-search
pytest tests/test_search.py -v
```

Expected: `7 passed in 0.XXs`

- [ ] **Step 5: Commit**

```bash
cd clip-person-search
git add search.py tests/test_search.py
git commit -m "feat: add similarity ranking with threshold and is_match for live mode"
```

---

## Task 4: detector.py — YOLOv8 Person Extraction

**Files:**
- Create: `clip-person-search/detector.py`
- Test: `clip-person-search/tests/test_detector.py`

- [ ] **Step 1: Write the failing tests**

Create `clip-person-search/tests/test_detector.py`:

```python
import numpy as np
import torch
from unittest.mock import MagicMock, patch


def _frame(h=200, w=200):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_returns_empty_list_when_no_detections():
    with patch("detector.model") as mock_model:
        mock_result = MagicMock()
        mock_result.boxes = []
        mock_model.return_value = [mock_result]
        from detector import extract_persons
        assert extract_persons(_frame()) == []


def test_returns_crop_and_box_for_valid_detection():
    frame = _frame(200, 200)
    frame[10:100, 10:80] = [0, 0, 255]
    with patch("detector.model") as mock_model:
        mock_box = MagicMock()
        mock_box.xyxy = [torch.tensor([10.0, 10.0, 80.0, 100.0])]
        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_model.return_value = [mock_result]
        from detector import extract_persons
        persons = extract_persons(frame)
    assert len(persons) == 1
    crop, box = persons[0]
    assert crop.shape[0] > 0 and crop.shape[1] > 0
    assert box == (10, 10, 80, 100)


def test_filters_zero_area_crops():
    with patch("detector.model") as mock_model:
        mock_box = MagicMock()
        mock_box.xyxy = [torch.tensor([10.0, 50.0, 80.0, 50.0])]
        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_model.return_value = [mock_result]
        from detector import extract_persons
        assert extract_persons(_frame()) == []
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd clip-person-search
pytest tests/test_detector.py -v
```

Expected: `ImportError: No module named 'detector'`

- [ ] **Step 3: Implement detector.py**

Create `clip-person-search/detector.py`:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")


def extract_persons(frame):
    """Returns list of (crop, box) for every person detected in frame."""
    results = model(frame, classes=[0], verbose=False)
    persons = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            persons.append((crop, (x1, y1, x2, y2)))
    return persons
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd clip-person-search
pytest tests/test_detector.py -v
```

Expected: `3 passed in 0.XXs`

- [ ] **Step 5: Commit**

```bash
cd clip-person-search
git add detector.py tests/test_detector.py
git commit -m "feat: add YOLOv8 person extraction in detector.py"
```

---

## Task 5: embedder.py — CLIP Image + Text Embeddings

**Files:**
- Create: `clip-person-search/embedder.py`
- Test: `clip-person-search/tests/test_embedder.py`

- [ ] **Step 1: Write the failing tests**

Create `clip-person-search/tests/test_embedder.py`:

```python
import numpy as np
import torch
from unittest.mock import patch


def _bgr_crop(h=64, w=64):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_embed_image_returns_tensor():
    with patch("embedder.model") as mock_model, \
         patch("embedder.preprocess") as mock_preprocess:
        mock_preprocess.return_value = torch.zeros(3, 224, 224)
        mock_model.encode_image.return_value = torch.zeros(1, 512)
        from embedder import embed_image
        result = embed_image(_bgr_crop())
    assert isinstance(result, torch.Tensor)
    mock_model.encode_image.assert_called_once()


def test_embed_image_accepts_bgr_without_error():
    crop = np.zeros((64, 64, 3), dtype=np.uint8)
    crop[:, :, 2] = 255
    with patch("embedder.model") as mock_model, \
         patch("embedder.preprocess") as mock_preprocess:
        mock_preprocess.return_value = torch.zeros(3, 224, 224)
        mock_model.encode_image.return_value = torch.zeros(1, 512)
        from embedder import embed_image
        embed_image(crop)  # must not raise


def test_embed_text_returns_tensor():
    with patch("embedder.model") as mock_model, \
         patch("embedder.clip") as mock_clip:
        mock_clip.tokenize.return_value = torch.zeros(1, 77, dtype=torch.long)
        mock_model.encode_text.return_value = torch.zeros(1, 512)
        from embedder import embed_text
        result = embed_text("red shirt man")
    assert isinstance(result, torch.Tensor)
    mock_clip.tokenize.assert_called_once_with(["red shirt man"])
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd clip-person-search
pytest tests/test_embedder.py -v
```

Expected: `ImportError: No module named 'embedder'`

- [ ] **Step 3: Implement embedder.py**

Create `clip-person-search/embedder.py`:

```python
import clip
import torch
from PIL import Image
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def embed_image(crop_bgr):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(crop_rgb)
    tensor = preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        return model.encode_image(tensor)


def embed_text(query):
    tokens = clip.tokenize([query]).to(device)
    with torch.no_grad():
        return model.encode_text(tokens)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd clip-person-search
pytest tests/test_embedder.py -v
```

Expected: `3 passed in 0.XXs`

- [ ] **Step 5: Commit**

```bash
cd clip-person-search
git add embedder.py tests/test_embedder.py
git commit -m "feat: add CLIP image and text embedding in embedder.py"
```

---

## Task 6: tracker.py — IoU Tracking + Sharpness-Gated Re-embedding

`PersonTracker` is the answer to "how often do we embed the same person?": at most once per quality improvement. When the same person is detected again (IoU ≥ 0.4 with a cached box), we compute the new crop's Laplacian variance. If it's higher than the stored sharpness, we call `embed_fn` and update the cache. Otherwise we return the existing embedding unchanged. Tracks expire after `max_age` unsampled frames so a person who leaves and returns is treated as new.

`sharpness` and `_iou` are pure functions — easy to unit test without any ML mocking.

**Files:**
- Create: `clip-person-search/tracker.py`
- Test: `clip-person-search/tests/test_tracker.py`

- [ ] **Step 1: Write the failing tests**

Create `clip-person-search/tests/test_tracker.py`:

```python
import numpy as np
import torch
from tracker import sharpness, PersonTracker


def _uniform(value=128, h=64, w=64):
    """Uniform BGR crop — near-zero Laplacian variance."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def _sharp():
    """Crop with a hard horizontal edge — high Laplacian variance."""
    crop = np.zeros((64, 64, 3), dtype=np.uint8)
    crop[:32, :] = 255
    return crop


def _embed_fn(crop):
    return torch.ones(1, 512)


# --- sharpness ---

def test_uniform_crop_has_low_sharpness():
    assert sharpness(_uniform()) < 1.0


def test_edge_crop_has_higher_sharpness_than_uniform():
    assert sharpness(_sharp()) > sharpness(_uniform())


# --- PersonTracker ---

def test_new_person_always_embeds():
    tracker = PersonTracker()
    calls = []
    def embed_fn(crop):
        calls.append(1)
        return torch.ones(1, 512)

    emb, re_embedded = tracker.update((10, 10, 50, 80), _sharp(), frame_count=0, embed_fn=embed_fn)
    assert re_embedded is True
    assert len(calls) == 1
    assert isinstance(emb, torch.Tensor)


def test_same_person_blurrier_crop_reuses_cached_embedding():
    tracker = PersonTracker()
    calls = []
    def embed_fn(crop):
        calls.append(1)
        return torch.ones(1, 512)

    box = (10, 10, 50, 80)
    tracker.update(box, _sharp(),   frame_count=0, embed_fn=embed_fn)  # sharp first
    _, re_embedded = tracker.update(box, _uniform(), frame_count=1, embed_fn=embed_fn)

    assert re_embedded is False
    assert len(calls) == 1  # embed_fn called only once


def test_same_person_sharper_crop_reembeds():
    tracker = PersonTracker()
    calls = []
    def embed_fn(crop):
        calls.append(1)
        return torch.ones(1, 512)

    box = (10, 10, 50, 80)
    tracker.update(box, _uniform(), frame_count=0, embed_fn=embed_fn)  # blurry first
    _, re_embedded = tracker.update(box, _sharp(),   frame_count=1, embed_fn=embed_fn)

    assert re_embedded is True
    assert len(calls) == 2


def test_expired_track_treated_as_new_person():
    tracker = PersonTracker(max_age=5)
    calls = []
    def embed_fn(crop):
        calls.append(1)
        return torch.ones(1, 512)

    box = (10, 10, 50, 80)
    tracker.update(box, _uniform(), frame_count=0,   embed_fn=embed_fn)
    _, re_embedded = tracker.update(box, _uniform(), frame_count=100, embed_fn=embed_fn)

    assert re_embedded is True
    assert len(calls) == 2  # treated as new person after expiry


def test_non_overlapping_boxes_are_independent_persons():
    tracker = PersonTracker()
    calls = []
    def embed_fn(crop):
        calls.append(1)
        return torch.ones(1, 512)

    tracker.update((0,   0,  50, 80), _uniform(), frame_count=0, embed_fn=embed_fn)
    tracker.update((200, 0, 250, 80), _uniform(), frame_count=0, embed_fn=embed_fn)

    assert len(calls) == 2
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd clip-person-search
pytest tests/test_tracker.py -v
```

Expected: `ImportError: No module named 'tracker'`

- [ ] **Step 3: Implement tracker.py**

Create `clip-person-search/tracker.py`:

```python
import cv2
import numpy as np

IOU_THRESHOLD = 0.4
MAX_AGE = 30  # sampled frames before a track expires


def sharpness(crop_bgr):
    """Laplacian variance of the crop. Higher = sharper image."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


class PersonTracker:
    """
    Tracks persons across frames by bounding box overlap.
    Calls embed_fn only when the new crop is sharper than the cached one.
    """

    def __init__(self, iou_threshold=IOU_THRESHOLD, max_age=MAX_AGE):
        self.tracks = []  # [{box, embedding, sharpness, last_frame}]
        self.iou_threshold = iou_threshold
        self.max_age = max_age

    def update(self, box, crop, frame_count, embed_fn):
        """
        Returns (embedding, re_embedded: bool).
        re_embedded=True  → embed_fn was called (new person or sharper crop).
        re_embedded=False → cached embedding returned; skip is_match / candidate.
        """
        self._expire(frame_count)

        for track in self.tracks:
            if _iou(box, track["box"]) >= self.iou_threshold:
                track["box"] = box
                track["last_frame"] = frame_count
                new_sharpness = sharpness(crop)
                if new_sharpness > track["sharpness"]:
                    track["embedding"] = embed_fn(crop)
                    track["sharpness"] = new_sharpness
                    return track["embedding"], True
                return track["embedding"], False

        # New person
        emb = embed_fn(crop)
        self.tracks.append({
            "box": box,
            "embedding": emb,
            "sharpness": sharpness(crop),
            "last_frame": frame_count,
        })
        return emb, True

    def _expire(self, current_frame):
        self.tracks = [
            t for t in self.tracks
            if current_frame - t["last_frame"] <= self.max_age
        ]
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd clip-person-search
pytest tests/test_tracker.py -v
```

Expected:
```
tests/test_tracker.py::test_uniform_crop_has_low_sharpness PASSED
tests/test_tracker.py::test_edge_crop_has_higher_sharpness_than_uniform PASSED
tests/test_tracker.py::test_new_person_always_embeds PASSED
tests/test_tracker.py::test_same_person_blurrier_crop_reuses_cached_embedding PASSED
tests/test_tracker.py::test_same_person_sharper_crop_reembeds PASSED
tests/test_tracker.py::test_expired_track_treated_as_new_person PASSED
tests/test_tracker.py::test_non_overlapping_boxes_are_independent_persons PASSED
7 passed in 0.XXs
```

- [ ] **Step 5: Commit**

```bash
cd clip-person-search
git add tracker.py tests/test_tracker.py
git commit -m "feat: add IoU tracker with sharpness-gated re-embedding"
```

---

## Task 7: main.py — Two Modes, Timestamps, draw_result

Public functions:
- `timestamp_from_frame(frame_count, fps) -> str`
- `draw_result(frame, box, score, query, timestamp) -> np.ndarray`
- `run_video(source, query_emb, query)` — batch mode; `PersonTracker` deduplicates embeddings; only appends candidate on re-embed
- `run_live(query_emb, query)` — streaming mode; only calls `is_match` on re-embed
- `main()` — calls `prompt_query()`, dispatches to `run_video` or `run_live`

**Files:**
- Create: `clip-person-search/main.py`
- Test: `clip-person-search/tests/test_main.py`

- [ ] **Step 1: Write the failing tests**

Create `clip-person-search/tests/test_main.py`:

```python
import numpy as np
import pytest


# --- timestamp_from_frame ---

def test_timestamp_zero():
    from main import timestamp_from_frame
    assert timestamp_from_frame(0, 30.0) == "00:00:00"


def test_timestamp_one_minute():
    from main import timestamp_from_frame
    assert timestamp_from_frame(1800, 30.0) == "00:01:00"


def test_timestamp_one_hour():
    from main import timestamp_from_frame
    assert timestamp_from_frame(108000, 30.0) == "01:00:00"


def test_timestamp_mixed():
    from main import timestamp_from_frame
    # 25 fps × 3725 frames = 149 s = 00:02:29
    assert timestamp_from_frame(3725, 25.0) == "00:02:29"


# --- draw_result ---

def test_draw_result_adds_green_rectangle():
    from main import draw_result
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    result = draw_result(frame, (10, 10, 100, 150), 0.85, "red shirt", "00:01:23")
    assert result is not None
    assert result.shape == (200, 300, 3)
    assert result[:, :, 1].max() == 255


def test_draw_result_handles_box_at_frame_edge():
    from main import draw_result
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    draw_result(frame, (0, 0, 100, 100), 0.5, "hat", "00:00:00")


def test_draw_result_label_clips_negative_y():
    from main import draw_result
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    draw_result(frame, (10, 0, 100, 80), 0.72, "blue jacket", "00:00:10")
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd clip-person-search
pytest tests/test_main.py -v
```

Expected: `ImportError: No module named 'main'`

- [ ] **Step 3: Implement main.py**

Create `clip-person-search/main.py`:

```python
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
    label = f"MATCH {score:.2f} | {query} | {timestamp}"
    cv2.putText(frame, label, (x1, max(y1 - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


def run_video(source, query_emb, query):
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    candidates = []
    frame_count = 0
    tracker = PersonTracker()

    print("Processing video...")

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
        print("No persons detected.")
        return

    print(f"Searching among {len(candidates)} detections...")
    result = find_best_match(query_emb, candidates)

    if result is None:
        print("No match found above similarity threshold.")
        return

    frame, box, score, timestamp = result
    output = draw_result(frame, box, score, query, timestamp)
    cv2.imshow(f"Best match — score: {score:.2f} @ {timestamp}", output)
    cv2.imwrite("result.jpg", output)
    print(f"Match! Score: {score:.2f} at {timestamp} — saved to result.jpg")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_live(query_emb, query):
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
                    score, matched = is_match(query_emb, img_emb)
                    if matched:
                        output = draw_result(frame.copy(), box, score, query, timestamp)
                        cv2.imshow("MATCH FOUND", output)
                        filename = f"match_{timestamp.replace(':', '')}.jpg"
                        cv2.imwrite(filename, output)
                        print(f"Match! Score: {score:.2f} at {timestamp} — saved to {filename}")

        cv2.imshow("Live Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    source = input("Video file path (or press Enter for live webcam): ").strip()
    query = prompt_query()
    query_emb = embed_text(query)

    if source:
        run_video(source, query_emb, query)
    else:
        run_live(query_emb, query)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd clip-person-search
pytest tests/test_main.py -v
```

Expected: `7 passed in 0.XXs`

- [ ] **Step 5: Commit**

```bash
cd clip-person-search
git add main.py tests/test_main.py
git commit -m "feat: add pipeline with PersonTracker, timestamps, and structured query input"
```

---

## Task 8: Full Test Suite + Smoke Run

**Files:** No new files.

- [ ] **Step 1: Run full test suite**

```bash
cd clip-person-search
pytest tests/ -v
```

Expected (38 tests):
```
tests/test_query_builder.py  11 passed
tests/test_search.py          7 passed
tests/test_detector.py        3 passed
tests/test_embedder.py        3 passed
tests/test_tracker.py         7 passed
tests/test_main.py            7 passed
38 passed in 0.XXs
```

- [ ] **Step 2: Smoke-test video mode**

```bash
cd clip-person-search
python main.py
# Video file path: your_video.mp4
# Upper clothing color: red
# Lower clothing color: black
# Has backpack? y   Has hat? n   Extra: beard
# Searching for: "person with red shirt and black pants and backpack and beard"
# Processing video...
# Searching among N detections...
# → result.jpg, or "No match found above similarity threshold."
```

- [ ] **Step 3: Smoke-test live mode**

```bash
cd clip-person-search
python main.py
# Video file path: (press Enter)
# Fill in description prompts
# → Live feed opens. CLIP is called only when a person appears
#   for the first time or a sharper frame arrives.
#   match_HHMMSS.jpg saved on each match. Press Q to quit.
```

- [ ] **Step 4: Final commit**

```bash
cd clip-person-search
git add -A
git commit -m "feat: complete clip-person-search — 38 tests passing"
```

---

## Threshold Tuning Reference

| Situation | Adjustment |
|-----------|-----------|
| Too many false positives | Raise `SIMILARITY_THRESHOLD` in `search.py` (try 0.25) |
| Missing obvious matches | Lower `SIMILARITY_THRESHOLD` (try 0.15) |
| Same person embedded too often | Raise `IOU_THRESHOLD` in `tracker.py` (try 0.5) |
| Different people collapsing into one track | Lower `IOU_THRESHOLD` (try 0.3) |
| Person reappears but track already expired | Lower `MAX_AGE` in `tracker.py` (try 15) |
| Too slow on CPU | Raise `SAMPLE_EVERY` in `main.py` (try 30) |

## Runtime Notes

- CLIP is called only when: (a) a person is seen for the first time, or (b) a sharper crop of the same person arrives. A person standing still gets exactly one CLIP call.
- Live mode saves `match_HHMMSS.jpg` per match. Multiple matches within the same second overwrite each other — add a frame counter suffix if needed.
- To use a non-default camera, change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `run_live`.
