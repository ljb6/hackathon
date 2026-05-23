# Streamlit UI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Streamlit web app that lets an investigator select cameras from `videos/`, describe a suspect, and view all detections ranked by CLIP relative similarity score.

**Architecture:** Single `app.py` at repo root adds `clip-person-search/` to `sys.path` and imports the existing pipeline. Sidebar for camera selection + query builder; main area for ranked results split into above-threshold cards and a below-threshold expander. A new public `collect_all_candidates` function in `main.py` returns all detections (not just the best) so the UI can rank and display every candidate. CLIP and YOLO are loaded once at startup via `@st.cache_resource`.

**Tech Stack:** Streamlit ≥ 1.35, clip-person-search pipeline (CLIP ViT-B/32, YOLOv8n, torch, opencv-python)

---

## File Map

| File | Change |
|------|--------|
| `clip-person-search/main.py` | Add `collect_all_candidates(source)` public function; refactor `_collect_candidates` to use it (DRY) |
| `app.py` | New — Streamlit UI at repo root |
| `requirements.txt` | New at repo root — `streamlit>=1.35` |

---

## Task 1: `collect_all_candidates` in main.py

The existing `_collect_candidates` mixes detection with best-match filtering. The UI needs all detections without filtering so it can rank them itself. Extract detection into a public function; keep `_collect_candidates` as a thin wrapper.

**Files:**
- Modify: `clip-person-search/main.py`

- [ ] **Step 1: Verify existing tests pass before touching main.py**

```bash
cd /Users/lucacarvalhojeo/development/Hackathon_Seg_Publica/hackathon/clip-person-search && python -m pytest tests/ -v
```

Expected: `42 passed`

- [ ] **Step 2: Add `collect_all_candidates` and refactor `_collect_candidates`**

Replace the `_collect_candidates` function in `clip-person-search/main.py` with this pair:

```python
def collect_all_candidates(source):
    """
    Process one video source. Returns (camera_label, candidates) where candidates is a
    list of (crop, box, frame, img_emb, timestamp) for every detected person, de-duplicated
    by the tracker and sampled every SAMPLE_EVERY frames.
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
    return camera_label, candidates


def _collect_candidates(source, query_emb, baseline_emb):
    """
    Process one video source. Returns (camera_label, result) where result is
    (frame, box, score, timestamp) or None if no match above threshold.
    """
    camera_label, candidates = collect_all_candidates(source)
    if not candidates:
        return camera_label, None
    result = find_best_match(query_emb, candidates, baseline_embedding=baseline_emb)
    return camera_label, result
```

- [ ] **Step 3: Verify all tests still pass after refactor**

```bash
cd /Users/lucacarvalhojeo/development/Hackathon_Seg_Publica/hackathon/clip-person-search && python -m pytest tests/ -v
```

Expected: `42 passed` — `_collect_candidates` behavior is unchanged, just delegating internally.

- [ ] **Step 4: Commit**

```bash
git -C /Users/lucacarvalhojeo/development/Hackathon_Seg_Publica/hackathon add clip-person-search/main.py
git -C /Users/lucacarvalhojeo/development/Hackathon_Seg_Publica/hackathon commit -m "feat: expose collect_all_candidates for UI integration"
```

---

## Task 2: Streamlit UI (`app.py`)

**Files:**
- Create: `app.py` (repo root)
- Create: `requirements.txt` (repo root)

- [ ] **Step 1: Install streamlit**

```bash
pip install streamlit
```

- [ ] **Step 2: Create `requirements.txt` at repo root**

```
streamlit>=1.35
```

- [ ] **Step 3: Create `app.py`**

Create `/Users/lucacarvalhojeo/development/Hackathon_Seg_Publica/hackathon/app.py`:

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "clip-person-search"))

import streamlit as st
from pathlib import Path
import cv2
import torch.nn.functional as F
from PIL import Image

from embedder import embed_text
from main import collect_all_candidates
from query_builder import compose_query
from search import RELATIVE_THRESHOLD

VIDEOS_DIR = Path(__file__).parent / "videos"
COLORS_UPPER = [
    "", "vermelho", "azul", "preto", "branco", "cinza",
    "verde", "amarelo", "laranja", "roxo", "marrom",
]
COLORS_LOWER = ["", "preto", "azul", "cinza", "branco", "marrom", "verde"]


@st.cache_resource(show_spinner="Carregando modelos CLIP e YOLO…")
def _baseline():
    return embed_text("a person")


def _score_and_rank(query_emb, tagged, baseline_emb):
    """
    tagged: list of (camera_label, crop, box, frame, img_emb, timestamp)
    Returns list of (score, camera_label, crop, box, frame, timestamp)
    sorted descending by relative score.
    """
    scored = []
    for camera_label, crop, box, frame, img_emb, timestamp in tagged:
        specific = F.cosine_similarity(query_emb, img_emb).item()
        generic = F.cosine_similarity(baseline_emb, img_emb).item()
        score = specific - generic
        scored.append((score, camera_label, crop, box, frame, timestamp))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def _crop_to_pil(crop_bgr):
    return Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Busca Visual de Suspeitos", layout="wide")
st.title("🔍 Busca Visual de Suspeitos")
st.caption("Selecione as câmeras, descreva o suspeito e clique em Buscar.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📹 Câmeras")
    video_files = sorted(VIDEOS_DIR.glob("*.mp4"))
    if not video_files:
        st.warning("Nenhum vídeo encontrado em videos/")
    selected = [
        f for f in video_files
        if st.checkbox(f.stem.replace("-", " ").title(), value=True, key=f.name)
    ]

    st.divider()
    st.header("🕵️ Suspeito")
    upper = st.selectbox("Cor — roupa superior", COLORS_UPPER)
    lower = st.selectbox("Cor — roupa inferior", COLORS_LOWER)
    backpack = st.checkbox("Tem mochila")
    hat = st.checkbox("Tem chapéu")
    extra = st.text_input("Outras características")

    query = compose_query(upper, lower, backpack, hat, extra)
    st.info(f'**Query:** "{query}"')

    search_btn = st.button(
        "🔍 Buscar",
        type="primary",
        use_container_width=True,
        disabled=not selected,
    )

# ── Search ────────────────────────────────────────────────────────────────────
if search_btn:
    baseline_emb = _baseline()
    query_emb = embed_text(query)
    tagged = []

    with st.status("Processando câmeras…", expanded=True) as status:
        for video_path in selected:
            st.write(f"⏳ Processando **{video_path.stem}**…")
            camera_label, candidates = collect_all_candidates(str(video_path))
            for c in candidates:
                tagged.append((camera_label, *c))
            st.write(f"✅ **{camera_label}** — {len(candidates)} detecções")

        status.update(label="Calculando scores…", state="running")
        results = _score_and_rank(query_emb, tagged, baseline_emb)
        status.update(
            label=f"Concluído — {len(results)} candidatos avaliados",
            state="complete",
        )

    st.session_state.results = results
    st.session_state.query = query

# ── Results ───────────────────────────────────────────────────────────────────
if "results" in st.session_state:
    results = st.session_state.results
    saved_query = st.session_state.get("query", "")

    above = [r for r in results if r[0] >= RELATIVE_THRESHOLD]
    below = [r for r in results if r[0] < RELATIVE_THRESHOLD]

    st.subheader(f'Resultados para: "{saved_query}"')
    m1, m2, m3 = st.columns(3)
    m1.metric("Correspondências", len(above))
    m2.metric("Abaixo do limiar", len(below))
    m3.metric("Total avaliados", len(results))

    if not above:
        st.info(
            "Nenhuma detecção acima do limiar de similaridade. "
            "Tente uma descrição diferente."
        )
    else:
        st.markdown("---")
        for rank, (score, camera_label, crop, box, frame, timestamp) in enumerate(above, 1):
            with st.container(border=True):
                col_img, col_info = st.columns([1, 4])
                with col_img:
                    st.image(_crop_to_pil(crop), width=110)
                with col_info:
                    st.markdown(f"**#{rank}** &nbsp; Score: `{score:.4f}`")
                    st.markdown(
                        f"📷 `{camera_label}`  &nbsp;  🕐 `{timestamp}`"
                    )

    if below:
        with st.expander(
            f"Ver {len(below)} detecções abaixo do limiar "
            f"(possíveis falsos negativos)"
        ):
            for score, camera_label, crop, box, frame, timestamp in below:
                col_img, col_info = st.columns([1, 5])
                with col_img:
                    st.image(_crop_to_pil(crop), width=80)
                with col_info:
                    st.caption(
                        f"Score `{score:.4f}` — `{camera_label}` @ `{timestamp}`"
                    )
```

- [ ] **Step 4: Run the app and verify the UI**

```bash
cd /Users/lucacarvalhojeo/development/Hackathon_Seg_Publica/hackathon && streamlit run app.py
```

Open `http://localhost:8501` and verify:

1. Sidebar shows video files from `videos/` as checkboxes (all checked by default)
2. Query dropdowns and checkboxes update the composed query string below them
3. Search button is disabled when no cameras are checked
4. Clicking Search → status box shows per-camera progress lines (`⏳` then `✅`)
5. After search: 3 metrics (Correspondências / Abaixo do limiar / Total avaliados)
6. Matches above threshold shown as bordered cards with crop image, score, camera, timestamp
7. Below-threshold detections visible in collapsed expander

- [ ] **Step 5: Commit**

```bash
git -C /Users/lucacarvalhojeo/development/Hackathon_Seg_Publica/hackathon add app.py requirements.txt
git -C /Users/lucacarvalhojeo/development/Hackathon_Seg_Publica/hackathon commit -m "feat: add Streamlit UI for visual suspect search"
```

---

## Threshold Tuning Reference

| Constant | Value | Location | Effect |
|---|---|---|---|
| `RELATIVE_THRESHOLD` | `0.02` | `clip-person-search/search.py` | Minimum delta to count as a match |
| `SAMPLE_EVERY` | `15` | `clip-person-search/main.py` | Frames skipped between samples |
| `IOU_THRESHOLD` | `0.4` | `clip-person-search/tracker.py` | Overlap for same-person tracking |
