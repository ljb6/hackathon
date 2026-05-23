# Hackathon Segurança Pública — Project Instructions

## Context

2-day public security hackathon. The goal is to make surveillance cameras queryable via natural language. An investigator describes a suspect ("man with red shirt and backpack") and the system returns photos with timestamp and camera location.

No auth, no real-time processing, no cross-camera re-ID. Demo-first scope.

## Architecture

```
process_videos.py   → reads video files, outputs deteccoes.json + crops/
query.py            → takes natural language description, calls LLM, returns matching crops
app.py              → Streamlit UI
```

No database. All state lives in `deteccoes.json` and the `crops/` folder.

## Tech stack

- **Detection:** YOLOv8n (`ultralytics`) — person detection + backpack/hat as COCO classes
- **Attribute extraction:** K-means on torso crop for clothing color
- **Storage:** `deteccoes.json` (flat list) + `crops/` (JPEG images)
- **Query:** LLM receives full JSON + user description, returns matching IDs
- **UI:** Streamlit
- **LLM:** Claude API (claude-haiku-4-5 for speed/cost)

## Data format

`deteccoes.json` — flat list of detection objects:

```json
[
  {
    "id": "cam1_0034",
    "camera": "Câmera 1 - Entrada",
    "timestamp": "00:00:34",
    "cor_superior": "vermelho",
    "cor_inferior": "preto",
    "tem_mochila": true,
    "tem_chapeu": false,
    "crop": "crops/cam1_0034.jpg"
  }
]
```

Colors are Portuguese plain text: "vermelho", "azul", "preto", "branco", "cinza", "verde", "amarelo", "laranja", "roxo", "marrom".

## Conventions

- 1 frame per second extracted from each video
- Crop is the bounding box of the person, saved as JPEG
- Torso crop = top 50% of the person bounding box, used for color extraction
- IDs follow pattern: `<camera_slug>_<seconds_zero_padded>_<detection_index>`
- `camera_id` in JSON matches the video filename (without extension)

## Out of scope

- Auth/login
- Real-time processing
- Multiple simultaneous queries
- Re-ID (linking same person across cameras)
- Error handling beyond basic try/except
- Mobile responsiveness
