"""
ASSIS PPE Detection — interactive demo.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from utils import summarize_results  # noqa: E402

WEIGHTS_PATH = REPO_ROOT / "models" / "best.pt"
FALLBACK_WEIGHTS = "yolov8n.pt"

st.set_page_config(page_title="ASSIS — PPE Compliance Demo", page_icon="🦺", layout="wide")


@st.cache_resource
def load_model():
    from ultralytics import YOLO

    if WEIGHTS_PATH.exists():
        return YOLO(str(WEIGHTS_PATH)), True
    return YOLO(FALLBACK_WEIGHTS), False


def main() -> None:
    st.title("🦺 ASSIS — PPE Compliance Detection")
    st.caption(
        "AI-Integrated Airport Safety & Security Intelligence System · "
        "Phase 1 MVP · Research demonstration only — not a regulatory tool"
    )

    model, is_trained = load_model()

    if not is_trained:
        st.warning(
            "No trained PPE weights at `models/best.pt` — running base YOLOv8 "
            "(detects people, not vests). Train via the Colab notebook first."
        )

    with st.sidebar:
        st.header("Settings")
        conf = st.slider("Confidence threshold", 0.1, 0.9, 0.4, 0.05)
        st.markdown("---")
        st.markdown(
            "**About**\n\nDetects high-visibility vest compliance in airport "
            "ramp environments. First module of the ASSIS research platform, "
            "aligned with FAA Part 139 SMS and TSA Part 1542 frameworks."
        )

    uploaded = st.file_uploader("Upload a ramp / worksite photo", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.info("Upload an image to run detection.")
        return

    image = Image.open(uploaded).convert("RGB")

    with st.spinner("Running detection..."):
        results = model(np.array(image), conf=conf)[0]

    annotated = results.plot()[:, :, ::-1]  # BGR -> RGB

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Detections")
        st.image(annotated, use_container_width=True)

    if is_trained:
        summary = summarize_results(results)

        st.markdown("---")
        st.subheader("Compliance Summary")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Personnel Detected", summary.total_people)
        m2.metric("Compliant (Vest)", summary.compliant)
        m3.metric("Violations", summary.violations)
        m4.metric("Compliance Rate", f"{summary.compliance_rate}%")

        if summary.violations > 0:
            st.error(f"⚠️ {summary.status}")
        elif summary.total_people > 0:
            st.success(f"✅ {summary.status}")
        else:
            st.info(summary.status)

        if summary.detections:
            with st.expander("Detection details"):
                st.dataframe(summary.detections, use_container_width=True)
    else:
        st.markdown("---")
        st.info("Showing base-model detections only. Train the PPE model to unlock compliance metrics.")


if __name__ == "__main__":
    main()
