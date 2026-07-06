import io
import os
import json
import datetime

import requests
import numpy as np
import pandas as pd
import cv2
import streamlit as st
from PIL import Image
from fpdf import FPDF

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Industrial Defect Detection",
    layout="wide"
)

st.title("🔍 Industrial Defect Detection")
st.caption("PatchCore + Classifier | MVTec AD (metal_nut, screw)")


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def overlay_heatmap(image: Image.Image, heatmap: list, alpha: float = 0.6, gamma: float = 2.2) -> np.ndarray:
    """
    Синий = норма, красный = только реально проблемная зона.

    JET-колормап даёт естественный переход синий → зелёный → жёлтый → красный
    по мере роста аномальности (в отличие от HOT, где даже фон уже красный).

    gamma > 1 дополнительно "прижимает" средние/низкие значения к синему краю,
    оставляя красный цвет только для по-настоящему высоких пиков аномалии.
    """
    hm = np.array(heatmap, dtype=np.float32)
    hm = cv2.resize(hm, (image.width, image.height))

    # Нормализация в диапазон [0, 1]
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)

    # Gamma-коррекция: усиливает контраст, фон остаётся синим,
    # красным подсвечиваются только явные пики
    hm = np.power(hm, gamma)

    hm = (hm * 255).astype(np.uint8)

    # COLORMAP_JET: синий (низкая аномальность) → красный (высокая аномальность)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    # cv2.applyColorMap всегда возвращает BGR, а изображение у нас в RGB —
    # без конвертации цвета инвертируются (красное становится синим и наоборот).
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)

    img_np = np.array(image.convert("RGB"))
    overlay = cv2.addWeighted(img_np, 1 - alpha, hm_color, alpha, 0)
    return overlay


def draw_bbox(image: Image.Image, bbox: dict | None, heatmap_shape: tuple[int, int] | None = None) -> np.ndarray:
    """Красный контур (без заливки) вокруг проблемной зоны"""
    img = np.array(image.convert("RGB")).copy()
    if not bbox:
        return img

    h, w = img.shape[:2]

    # Берём РЕАЛЬНЫЙ размер grid из heatmap, а не захардкоженные 7x7
    if heatmap_shape:
        grid_h, grid_w = heatmap_shape
    else:
        grid_h, grid_w = 7, 7  # запасной вариант, если shape не передан

    scale_x = w / grid_w
    scale_y = h / grid_h

    x1 = int(bbox.get("x1", 0) * scale_x)
    y1 = int(bbox.get("y1", 0) * scale_y)
    x2 = int(bbox.get("x2", 0) * scale_x)
    y2 = int(bbox.get("y2", 0) * scale_y)

    # Минимальный размер
    min_size = int(min(w, h) * 0.18)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if (x2 - x1) < min_size:
        cx = (x1 + x2) // 2
        x1 = max(0, cx - min_size // 2)
        x2 = min(w, cx + min_size // 2)
    if (y2 - y1) < min_size:
        cy = (y1 + y2) // 2
        y1 = max(0, cy - min_size // 2)
        y2 = min(h, cy + min_size // 2)

    # Только контур, без заливки.
    # img здесь в порядке RGB (из image.convert("RGB")), поэтому для красного
    # цвета нужен (255, 0, 0), а не (0, 0, 255) — последнее дало бы синий.
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 4)

    return img


def load_model_comparison() -> pd.DataFrame | None:
    paths = [
        "runs/final_model_comparison.csv",
        "runs/metrics/model_comparison.csv",
    ]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None


# ─────────────────────────────────────────
# PDF GENERATION
# ─────────────────────────────────────────
def generate_pdf(
    image: Image.Image,
    result: dict,
    heatmap_img: np.ndarray,
    bbox_img: np.ndarray,
    comparison_df: pd.DataFrame | None,
) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Title ──
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Industrial Defect Detection - Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(6)

    # ── Prediction result ──
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Prediction Result", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)

    label_text = result.get("decision", "N/A").upper()
    pdf.set_text_color(200, 0, 0) if label_text == "DEFECT" else pdf.set_text_color(0, 150, 0)
    pdf.cell(0, 7, f"Decision: {label_text}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)

    pdf.cell(0, 7, f"Classifier probability (defect): {result.get('classifier_prob', 0):.4f}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"PatchCore anomaly score: {result.get('patchcore_score', 0):.4f}", new_x="LMARGIN", new_y="NEXT")

    bbox = result.get("bbox")
    if bbox:
        pdf.cell(0, 7,
            f"Defect bbox: x1={bbox['x1']} y1={bbox['y1']} x2={bbox['x2']} y2={bbox['y2']}",
            new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── Save images to temp buffers ──
    def pil_to_tmp(img_array: np.ndarray, name: str) -> str:
        path = f"/tmp/{name}.png"
        Image.fromarray(img_array).save(path)
        return path

    orig_path = "/tmp/pdf_orig.png"
    image.resize((224, 224)).save(orig_path)
    heatmap_path = pil_to_tmp(heatmap_img, "pdf_heatmap")
    bbox_path = pil_to_tmp(bbox_img, "pdf_bbox")

    # ── Images side by side ──
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Visualizations", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    img_w = 58
    x_positions = [10, 75, 140]
    captions = ["Original", "Heatmap", "BBox Localization"]
    paths = [orig_path, heatmap_path, bbox_path]

    y_start = pdf.get_y()
    for x_pos, caption, img_path in zip(x_positions, captions, paths):
        pdf.image(img_path, x=x_pos, y=y_start, w=img_w)
        pdf.set_xy(x_pos, y_start + img_w + 1)
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(img_w, 5, caption, align="C")

    pdf.set_y(y_start + img_w + 10)
    pdf.ln(4)

    # ── Model comparison table ──
    if comparison_df is not None:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Model Comparison", new_x="LMARGIN", new_y="NEXT")

        cols = list(comparison_df.columns)
        col_w = min(180 // len(cols), 40)

        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(220, 220, 220)
        for col in cols:
            pdf.cell(col_w, 7, str(col)[:18], border=1, fill=True)
        pdf.ln()

        pdf.set_font("Helvetica", "", 8)
        for _, row in comparison_df.iterrows():
            for col in cols:
                val = str(row[col])
                if len(val) > 18:
                    val = val[:15] + "..."
                pdf.cell(col_w, 6, val, border=1)
            pdf.ln()

    # ── Return bytes ──
    return bytes(pdf.output())


# ─────────────────────────────────────────
# SESSION STATE — история запросов
# ─────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ─────────────────────────────────────────
# SIDEBAR — история
# ─────────────────────────────────────────
with st.sidebar:
    st.header("📋 Session History")
    if st.session_state.history:
        for i, h in enumerate(reversed(st.session_state.history)):
            icon = "🔴" if h["decision"] == "defect" else "🟢"
            st.write(f"{icon} **{h['filename']}** — {h['decision']} ({h['time']})")
    else:
        st.caption("No predictions yet.")

    if st.session_state.history:
        history_json = json.dumps(st.session_state.history, indent=2)
        st.download_button(
            "💾 Download history (JSON)",
            data=history_json,
            file_name="session_history.json",
            mime="application/json"
        )


# ─────────────────────────────────────────
# MAIN — upload + predict
# ─────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload image (PNG / JPG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image, caption="Input image", use_container_width=True)

    # ── Call FastAPI ──
    with st.spinner("Running inference..."):
        try:
            uploaded_file.seek(0)
            response = requests.post(
                f"{API_URL}/predict",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "image/png")},
                timeout=60
            )
            response.raise_for_status()
            result = response.json()

        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Is FastAPI running?")
            st.stop()
        except Exception as e:
            st.error(f"❌ API error: {e}")
            st.stop()

    # ── Results ──
    with col2:
        decision = result.get("decision", "N/A")
        color = "🔴" if decision == "defect" else "🟢"

        st.subheader(f"{color} Decision: **{decision.upper()}**")
        st.metric("Classifier probability", f"{result.get('classifier_prob', 0):.4f}")
        st.metric("PatchCore score", f"{result.get('patchcore_score', 0):.4f}")

        bbox = result.get("bbox")
        if bbox:
            st.caption(
                f"Defect bbox → x1={bbox['x1']} y1={bbox['y1']} "
                f"x2={bbox['x2']} y2={bbox['y2']}"
            )

    st.divider()

    # ── Visualizations ──
    heatmap_data = result["heatmap"]
    grid_h = len(heatmap_data)
    grid_w = len(heatmap_data[0]) if grid_h else 7

    heatmap_img = overlay_heatmap(image, heatmap_data)
    bbox_img = draw_bbox(image, result.get("bbox"), heatmap_shape=(grid_h, grid_w))

    v1, v2, v3 = st.columns(3)
    with v1:
        st.image(image, caption="Original", use_container_width=True)
    with v2:
        st.image(heatmap_img, caption="🔥 Anomaly Heatmap", use_container_width=True)
    with v3:
        st.image(bbox_img, caption="📦 Defect Localization", use_container_width=True)

    # ── Save to history ──
    st.session_state.history.append({
        "filename": uploaded_file.name,
        "decision": decision,
        "classifier_prob": result.get("classifier_prob"),
        "patchcore_score": result.get("patchcore_score"),
        "time": datetime.datetime.now().strftime("%H:%M:%S")
    })

    # ── PDF report ──
    st.divider()
    st.subheader("📄 Download Report")

    comparison_df = load_model_comparison()

    pdf_bytes = generate_pdf(
        image=image,
        result=result,
        heatmap_img=heatmap_img,
        bbox_img=bbox_img,
        comparison_df=comparison_df,
    )

    st.download_button(
        label="⬇️ Download PDF Report",
        data=pdf_bytes,
        file_name=f"report_{uploaded_file.name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf"
    )
