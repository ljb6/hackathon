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

# UI labels in Portuguese → CLIP English tokens
PT_TO_EN = {
    "vermelho": "red",
    "azul": "blue",
    "preto": "black",
    "branco": "white",
    "cinza": "gray",
    "verde": "green",
    "amarelo": "yellow",
    "laranja": "orange",
    "roxo": "purple",
    "marrom": "brown",
}

COLORS_UPPER_PT = ["", "vermelho", "azul", "preto", "branco", "cinza", "verde", "amarelo", "laranja", "roxo", "marrom"]
COLORS_LOWER_PT = ["", "preto", "azul", "cinza", "branco", "marrom", "verde"]


@st.cache_resource(show_spinner="Preparando referência de busca…")
def _baseline():
    return embed_text("a person")


def _score_and_rank(query_emb, tagged, baseline_emb):
    """
    tagged: list of (camera_label, crop, img_emb, timestamp)
    Returns list of (score, camera_label, crop, timestamp) sorted descending.
    """
    scored = []
    for camera_label, crop, img_emb, timestamp in tagged:
        specific = F.cosine_similarity(query_emb, img_emb).item()
        generic = F.cosine_similarity(baseline_emb, img_emb).item()
        score = specific - generic
        scored.append((score, camera_label, crop, timestamp))
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
        st.stop()
    selected = [
        f for f in video_files
        if st.checkbox(f.stem.replace("-", " ").title(), value=True, key=f.name)
    ]

    st.divider()
    st.header("🕵️ Suspeito")
    upper_pt = st.selectbox("Cor — roupa superior", COLORS_UPPER_PT)
    lower_pt = st.selectbox("Cor — roupa inferior", COLORS_LOWER_PT)
    backpack = st.checkbox("Tem mochila")
    hat = st.checkbox("Tem chapéu")
    extra = st.text_input("Outras características (em inglês)")

    # Translate PT colors to English for CLIP
    upper_en = PT_TO_EN.get(upper_pt, "")
    lower_en = PT_TO_EN.get(lower_pt, "")
    query = compose_query(upper_en, lower_en, backpack, hat, extra)
    st.info(f'**Query CLIP:** "{query}"')

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
    tagged = []  # (camera_label, crop, img_emb, timestamp)

    with st.status("Processando câmeras…", expanded=True) as status:
        for video_path in selected:
            st.write(f"⏳ Processando **{video_path.stem}**…")
            try:
                camera_label, candidates = collect_all_candidates(str(video_path))
            except Exception as e:
                st.error(f"Erro ao processar {video_path.stem}: {e}")
                continue

            if not candidates:
                st.warning(f"⚠️ {video_path.stem} — nenhuma pessoa detectada (verifique o arquivo)")
                continue

            for crop, box, frame, img_emb, timestamp in candidates:
                tagged.append((camera_label, crop, img_emb, timestamp))
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
        for rank, (score, camera_label, crop, timestamp) in enumerate(above, 1):
            with st.container(border=True):
                col_img, col_info = st.columns([1, 4])
                with col_img:
                    st.image(_crop_to_pil(crop), width=110)
                with col_info:
                    st.markdown(f"**#{rank}** &nbsp; Score: `{score:.4f}`")
                    st.markdown(f"📷 `{camera_label}`  &nbsp;  🕐 `{timestamp}`")

    if below:
        with st.expander(
            f"Ver {len(below)} detecções abaixo do limiar "
            f"(possíveis falsos negativos)"
        ):
            for score, camera_label, crop, timestamp in below:
                col_img, col_info = st.columns([1, 5])
                with col_img:
                    st.image(_crop_to_pil(crop), width=80)
                with col_info:
                    st.caption(
                        f"Score `{score:.4f}` — `{camera_label}` @ `{timestamp}`"
                    )
