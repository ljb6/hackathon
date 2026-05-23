# Image Upload Search — Design Spec

**Date:** 2026-05-23  
**Status:** Approved

## Problem

The current UI only accepts structured form inputs (color dropdowns, checkboxes) to describe a suspect. Investigators often have a photo and want to search directly from it without manually extracting attributes.

## Solution

Add an image uploader to the sidebar. When a photo is uploaded, embed it with CLIP and use that embedding as the query, replacing the text embedding. The rest of the scoring pipeline is unchanged.

## Architecture

Only `app.py` changes. All files in `clip-person-search/` are untouched.

`embed_image` already exists in `embedder.py` and accepts a BGR numpy array. The uploaded file is converted: bytes → PIL Image → numpy RGB → `cv2.cvtColor` → BGR.

## UI Layout

Sidebar under `🕵️ Suspeito`:

```
📎 Enviar foto do suspeito   [file uploader — jpg/png]

— if image uploaded —
  [thumbnail ~120px]
  ℹ️ "Busca por imagem enviada"

— if no image —
  [existing color dropdowns, checkboxes, text input]
  ℹ️ Query CLIP: "person with ..."
```

Camera selection and period filter sections are unchanged. Mode switches reactively on upload/remove.

## Data Flow

1. User uploads image via `st.file_uploader`
2. Bytes → `PIL.Image.open` → `np.array` (RGB) → `cv2.cvtColor(BGR)`
3. At search time: `query_emb = embed_image(bgr)` instead of `embed_text(query)`
4. `_score_and_rank(query_emb, tagged, baseline_emb)` — identical to text path

## Error Handling

- Corrupt upload: try/except around PIL decode → `st.error`, fall back to form mode
- Empty array after conversion: check `bgr.size > 0` → `st.error` if failed
- No file + empty form: existing fallback (`compose_query` returns `"person"`)

## Files Changed

| File | Change |
|------|--------|
| `app.py` | Add file uploader, conditional mode logic, image→BGR conversion, swap `embed_text` for `embed_image` in image mode |

## Out of Scope

- Attribute extraction or form pre-filling from the uploaded image
- Showing which attributes were detected
- Multiple image uploads
