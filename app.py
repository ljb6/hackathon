import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "clip-person-search"))

import streamlit as st
from pathlib import Path
from datetime import time as dtime, date as ddate, datetime
import cv2
from PIL import Image

from main import collect_all_candidates
from query_builder import compose_query
from search import RELATIVE_THRESHOLD
from multi_scorer import MultiAttributeScorer, ATTRIBUTE_THRESHOLD
from query_expander import expand_query

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


@st.cache_resource(show_spinner="Carregando scorer de atributos…")
def _scorer():
    return MultiAttributeScorer()


def _score_and_rank(tagged, scorer, upper_color="", lower_color="",
                    has_backpack=None, has_hat=None, extra=""):
    scored = []
    for camera_label, crop, img_emb, timestamp in tagged:
        total, breakdown = scorer.score(
            img_emb,
            upper_color=upper_color,
            lower_color=lower_color,
            has_backpack=has_backpack,
            has_hat=has_hat,
            extra=extra,
        )
        scored.append((total, breakdown, camera_label, crop, timestamp))
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
    extra = st.text_input("Outras características")

    upper_en = PT_TO_EN.get(upper_pt, "")
    lower_en = PT_TO_EN.get(lower_pt, "")
    query = compose_query(upper_en, lower_en, backpack, hat, extra)
    st.info(f'**Query CLIP:** "{query}"')
    if extra:
        st.caption("📝 Outras características serão expandidas automaticamente ao buscar")

    st.divider()
    st.header("📅 Período")
    filter_date = st.date_input("Data", value=ddate.today())
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        filter_from = st.time_input("Das", value=dtime(0, 0))
    with col_t2:
        filter_to = st.time_input("Até", value=dtime(23, 59))

    search_btn = st.button(
        "🔍 Buscar",
        type="primary",
        use_container_width=True,
        disabled=not selected,
    )

# ── Search ────────────────────────────────────────────────────────────────────
if search_btn:
    scorer = _scorer()
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

        status.update(label="Expandindo query e calculando scores…", state="running")
        expanded_extra = expand_query(extra) if extra else ""

        has_structured = any([upper_pt, lower_pt, backpack, hat])
        threshold = ATTRIBUTE_THRESHOLD if has_structured else RELATIVE_THRESHOLD

        results = _score_and_rank(
            tagged,
            scorer,
            upper_color=upper_en,
            lower_color=lower_en,
            has_backpack=backpack if backpack else None,
            has_hat=hat if hat else None,
            extra=expanded_extra,
        )
        status.update(
            label=f"Concluído — {len(results)} candidatos avaliados",
            state="complete",
        )

    if expanded_extra and expanded_extra != extra:
        st.caption(f'📝 Extra expandido: "{expanded_extra}"')

    st.session_state.results = results
    st.session_state.query = query
    st.session_state.threshold = threshold
    st.session_state.filter_date = filter_date
    st.session_state.filter_from = filter_from
    st.session_state.filter_to = filter_to

# ── Results ───────────────────────────────────────────────────────────────────
_ATTR_LABELS = {
    "upper_color": "👕 Superior",
    "lower_color": "👖 Inferior",
    "backpack":    "🎒 Mochila",
    "hat":         "🧢 Chapéu",
    "extra":       "📝 Extra",
}

if "results" in st.session_state:
    results = st.session_state.results
    saved_query = st.session_state.get("query", "")
    threshold = st.session_state.get("threshold", ATTRIBUTE_THRESHOLD)
    f_date = st.session_state.get("filter_date", ddate.today())
    f_from = st.session_state.get("filter_from", dtime(0, 0))
    f_to   = st.session_state.get("filter_to",   dtime(23, 59))

    above = [r for r in results if r[0] >= threshold]
    below = [r for r in results if r[0] < threshold]

    st.subheader(f'Resultados para: "{saved_query}"')
    st.caption(f"📅 {f_date.strftime('%d/%m/%Y')}  🕐 {f_from.strftime('%H:%M')} – {f_to.strftime('%H:%M')}")
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
        for rank, (score, breakdown, camera_label, crop, timestamp) in enumerate(above, 1):
            with st.container(border=True):
                col_img, col_info = st.columns([1, 4])
                with col_img:
                    st.image(_crop_to_pil(crop), width=110)
                with col_info:
                    st.markdown(f"**#{rank}** &nbsp; Score: `{score:.4f}`")
                    st.markdown(f"📷 `{camera_label}`  &nbsp;  🕐 `{f_date.strftime('%d/%m/%Y')} {timestamp}`")
                    if breakdown:
                        parts = [f"{_ATTR_LABELS.get(k, k)}: `{v:.0%}`" for k, v in breakdown.items()]
                        st.caption("  |  ".join(parts))

    if below:
        with st.expander(
            f"Ver {len(below)} detecções abaixo do limiar "
            f"(possíveis falsos negativos)"
        ):
            for score, breakdown, camera_label, crop, timestamp in below:
                col_img, col_info = st.columns([1, 5])
                with col_img:
                    st.image(_crop_to_pil(crop), width=80)
                with col_info:
                    st.caption(
                        f"Score `{score:.4f}` — `{camera_label}` @ `{f_date.strftime('%d/%m/%Y')} {timestamp}`"
                    )
