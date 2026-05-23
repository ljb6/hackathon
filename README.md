# Hackathon Segurança Pública — Busca Visual de Suspeitos

Search surveillance camera footage using natural language descriptions.

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install -r clip-person-search/requirements.txt
```

### 2. Start the app

```bash
python3 -m streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### 3. Use

1. Select which cameras to search (sidebar checkboxes)
2. Describe the suspect — upper/lower clothing color, backpack, hat, extra details
3. Click **Buscar**
4. Results appear ranked by similarity score — matches above threshold shown as cards, near-misses in the expander below

## CLI (no UI)

```bash
cd clip-person-search
python3 main.py
# Video file paths: ../videos/passageway1-c1.mp4,../videos/passageway1-c2.mp4
# Describe suspect interactively
```

## Project structure

```
app.py                    # Streamlit UI
clip-person-search/
  main.py                 # CLI entry point + video processing
  detector.py             # YOLOv8 person detection
  embedder.py             # CLIP text/image embedding
  search.py               # Relative similarity scoring
  tracker.py              # IoU-based person tracker
  query_builder.py        # Structured query composer
videos/                   # Camera footage (.mp4)
```
