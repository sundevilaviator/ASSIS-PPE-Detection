"""
ASSIS PPE Detection - interactive demo (v2, dual-model violation detection).

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

FINE_TUNED_WEIGHTS = REPO_ROOT / "models" / "best.pt"
BASE_WEIGHTS = "yolov8n.pt"

st.set_page_config(page_title="ASSIS - PPE Compliance Demo", page_icon="🦺", layout="wide")


@st.cache_resource
def load_models():
    from ultralytics import YOLO
    base_model = YOLO(BASE_WEIGHTS)
    if FINE_TUNED_WEIGHTS.exists():
        ppe_model = YOLO(str(FINE_TUNED_WEIGHTS))
        return base_model, ppe_model, True
    return base_model, None, False


def main() -> None:
    st.title("🦺 ASSIS - PPE Compliance Detection")
    st.caption(
        "AI-Integrated Airport Safety & Security Intelligence System · "
        "Phase 1 MVP · Dual-model person + PPE correlation"
    )

    base_model, ppe_model, is_trained = load_models()

    if not is_trained:
        st.warning("No trained PPE weights at `models/best.pt`.")
        return

    with st.sidebar:
        st.header("Settings")
        conf = st.slider("Confidence threshold", 0.1, 0.9, 0.4, 0.05)
        st.markdown("---")
        st.markdown(
            "**How this works**\n\n"
            "Every uploaded image is run through two models: a person "
            "detector and a fine-tuned PPE detector. The app checks "
            "whether each detected person overlaps with a vest detection "
            "- if not, that person is flagged as a violation."
        )

    uploaded = st.file_uploader("Upload a ramp / worksite photo", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.info("Upload an image to run detection.")
        return

    image = Image.open(uploaded).convert("RGB")
    image_np = np.array(image)

    with st.spinner("Running person detection + PPE detection..."):
        person_results = base_model(image_np, conf=conf, classes=[0])[0]
        ppe_results = ppe_model(image_np, conf=conf)[0]

    ppe_annotated = ppe_results.plot()[:, :, ::-1]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("PPE Detections")
        st.image(ppe_annotated, use_container_width=True)

    summary = summarize_results(person_results, ppe_results)

    st.markdown("---")
    st.subheader("Compliance Summary")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Personnel Detected", summary.total_people)
    m2.metric("Vest Compliant", summary.vest_compliant)
    m3.metric("Violations", summary.violations)
    m4.metric("Compliance Rate", f"{summary.compliance_rate}%")

    if summary.violations > 0:
        st.error(summary.status)
    elif summary.total_people > 0:
        st.success(summary.status)
    else:
        st.info(summary.status)

    if summary.raw_ppe_counts:
        with st.expander("Raw PPE item detections"):
            st.write(summary.raw_ppe_counts)

    if summary.people:
        with st.expander("Per-person compliance detail"):
            for i, person in enumerate(summary.people, 1):
                st.write(
                    f"Person {i} (conf {person.confidence}): "
                    f"vest={'YES' if person.has_vest else 'NO'}  "
                    f"helmet={'YES' if person.has_helmet else 'NO'}  "
                    f"gloves={'YES' if person.has_gloves else 'NO'}"
                )


if __name__ == "__main__":
    main()
