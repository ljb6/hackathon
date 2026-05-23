# Image Upload Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let investigators upload a photo of a suspect and search video feeds using CLIP image-to-image similarity, bypassing the text form entirely.

**Architecture:** A new `image_utils.py` module handles the file-bytes-to-BGR conversion. `app.py` gains a file uploader that, when used, calls `embed_image(bgr)` instead of `embed_text(query)` — the scoring pipeline is unchanged.

**Tech Stack:** Streamlit `st.file_uploader`, Pillow (already a transitive dep via CLIP), OpenCV (already used in `app.py`), CLIP `embed_image` (already in `embedder.py`)

---

## File Structure

| Path | Action | Responsibility |
|------|--------|----------------|
| `clip-person-search/image_utils.py` | Create | Convert uploaded file bytes → BGR numpy array |
| `clip-person-search/tests/test_image_utils.py` | Create | Unit tests for the conversion helper |
| `app.py` | Modify | File uploader UI, mode switching, image-to-query wiring |

---

### Task 1: Create `uploaded_file_to_bgr` with tests (TDD)

**Files:**
- Create: `clip-person-search/image_utils.py`
- Create: `clip-person-search/tests/test_image_utils.py`

- [ ] **Step 1: Write the failing tests**

Create `clip-person-search/tests/test_image_utils.py`:

```python
import io
import numpy as np
from PIL import Image


def _make_png_bytes(color_rgb, size=(10, 10)):
    arr = np.full((*size, 3), color_rgb, dtype=np.uint8)
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_returns_numpy_array():
    from image_utils import uploaded_file_to_bgr
    result = uploaded_file_to_bgr(_make_png_bytes([128, 64, 32]))
    assert isinstance(result, np.ndarray)


def test_shape_is_height_width_3():
    from image_utils import uploaded_file_to_bgr
    result = uploaded_file_to_bgr(_make_png_bytes([0, 0, 0], size=(20, 30)))
    assert result.shape == (20, 30, 3)


def test_channel_order_is_bgr():
    from image_utils import uploaded_file_to_bgr
    # Pure red RGB=[255,0,0] becomes BGR=[0,0,255]
    result = uploaded_file_to_bgr(_make_png_bytes([255, 0, 0]))
    assert result[5, 5, 0] < 10   # blue channel near zero
    assert result[5, 5, 2] > 245  # red channel near max


def test_nonempty_output():
    from image_utils import uploaded_file_to_bgr
    result = uploaded_file_to_bgr(_make_png_bytes([0, 0, 0]))
    assert result.size > 0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/lucca/hackathon/clip-person-search && python -m pytest tests/test_image_utils.py -v
```

Expected: `ModuleNotFoundError: No module named 'image_utils'`

- [ ] **Step 3: Implement `uploaded_file_to_bgr`**

Create `clip-person-search/image_utils.py`:

```python
import cv2
import numpy as np
from PIL import Image


def uploaded_file_to_bgr(uploaded_file) -> np.ndarray:
    pil = Image.open(uploaded_file).convert("RGB")
    rgb = np.array(pil)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /Users/lucca/hackathon/clip-person-search && python -m pytest tests/test_image_utils.py -v
```

Expected:
```
test_image_utils.py::test_returns_numpy_array PASSED
test_image_utils.py::test_shape_is_height_width_3 PASSED
test_image_utils.py::test_channel_order_is_bgr PASSED
test_image_utils.py::test_nonempty_output PASSED
4 passed
```

- [ ] **Step 5: Commit**

```bash
git add clip-person-search/image_utils.py clip-person-search/tests/test_image_utils.py
git commit -m "feat: add uploaded_file_to_bgr conversion helper"
```

---

### Task 2: Wire image upload into `app.py`

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Update imports at the top of `app.py`**

Replace:
```python
from embedder import embed_text
```
With:
```python
from embedder import embed_text, embed_image
from image_utils import uploaded_file_to_bgr
```

- [ ] **Step 2: Replace the sidebar Suspeito section**

Replace the entire block from `st.header("🕵️ Suspeito")` through `st.info(f'**Query CLIP:** "{query}"')` with:

```python
    st.header("🕵️ Suspeito")
    uploaded_file = st.file_uploader(
        "📎 Enviar foto do suspeito", type=["jpg", "jpeg", "png"]
    )

    suspect_bgr = None
    query = None

    if uploaded_file is not None:
        try:
            suspect_bgr = uploaded_file_to_bgr(uploaded_file)
            if suspect_bgr.size == 0:
                st.error("Imagem inválida. Tente outro arquivo.")
                suspect_bgr = None
        except Exception:
            st.error("Não foi possível ler a imagem. Tente outro arquivo.")

        if suspect_bgr is not None:
            st.image(_crop_to_pil(suspect_bgr), width=120)
            st.info("**Busca por imagem enviada**")
            query = "imagem enviada"
    else:
        upper_pt = st.selectbox("Cor — roupa superior", COLORS_UPPER_PT)
        lower_pt = st.selectbox("Cor — roupa inferior", COLORS_LOWER_PT)
        backpack = st.checkbox("Tem mochila")
        hat = st.checkbox("Tem chapéu")
        extra = st.text_input("Outras características (em inglês)")
        upper_en = PT_TO_EN.get(upper_pt, "")
        lower_en = PT_TO_EN.get(lower_pt, "")
        query = compose_query(upper_en, lower_en, backpack, hat, extra)
        st.info(f'**Query CLIP:** "{query}"')
```

- [ ] **Step 3: Update the search block to use image embedding when available**

Inside `if search_btn:`, replace:

```python
    baseline_emb = _baseline()
    query_emb = embed_text(query)
```

With:

```python
    if suspect_bgr is None and query is None:
        st.error("Imagem inválida. Tente outro arquivo.")
        st.stop()

    baseline_emb = _baseline()
    if suspect_bgr is not None:
        query_emb = embed_image(suspect_bgr)
    else:
        query_emb = embed_text(query)
```

- [ ] **Step 4: Verify the app starts without errors**

```bash
cd /Users/lucca/hackathon && streamlit run app.py
```

Expected: app opens in browser with no traceback, sidebar shows file uploader above the period section.

- [ ] **Step 5: Manual smoke test — text mode unchanged**

1. Leave the file uploader empty.
2. Select "vermelho" for upper color, click Buscar.
3. Expected: existing behavior — CLIP query `"person with red shirt"` shown, results appear.

- [ ] **Step 6: Manual smoke test — image mode**

1. Upload any JPEG photo of a person.
2. Confirm sidebar shows thumbnail + "Busca por imagem enviada".
3. Click Buscar.
4. Expected: results appear, result header shows `Resultados para: "imagem enviada"`.

- [ ] **Step 7: Commit**

```bash
git add app.py
git commit -m "feat: add image upload search using CLIP image embedding"
```
