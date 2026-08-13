"""
ASSIS PPE Detection - interactive demo (v3, dual-model violation detection).

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from utils import summarize_results  # noqa: E402
from conditions import fetch_metar, determine_requirements, is_valid_icao  # noqa: E402

FINE_TUNED_WEIGHTS = REPO_ROOT / "models" / "best.pt"
BASE_WEIGHTS = "yolov8n.pt"

st.set_page_config(page_title="ASSIS - PPE Compliance Demo", page_icon="🦺", layout="wide")


def weights_fingerprint(path: Path) -> str:
    """SHA-256 of the weights file.

    Used as a cache key so that replacing models/best.pt automatically
    invalidates the cached model instead of silently serving a stale one.
    """
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@st.cache_resource(show_spinner="Loading detection models...")
def load_models(fingerprint: str):
    """Load the person detector and the fine-tuned PPE detector.

    `fingerprint` is not used inside the function - it exists purely so
    Streamlit re-runs this loader whenever the weights file changes.
    """
    import torch

    # Patch torch.load to default to weights_only=False for this session.
    # Safe here because we are loading our own trained model file, not an
    # untrusted third-party checkpoint.
    _original_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _original_load(*args, **kwargs)

    torch.load = _patched_load

    from ultralytics import YOLO

    base_model = YOLO(BASE_WEIGHTS)
    if FINE_TUNED_WEIGHTS.exists():
        ppe_model = YOLO(str(FINE_TUNED_WEIGHTS))
        return base_model, ppe_model, True
    return base_model, None, False


def annotated_rgb(results) -> np.ndarray:
    """Return the annotated frame in RGB for display.

    The model is fed a BGR array (see `main`), so `plot()` returns BGR.
    Reversing the channels converts it back to RGB for Streamlit.
    """
    return results.plot()[:, :, ::-1]


def main() -> None:
    st.title("🦺 ASSIS - PPE Compliance Detection")
    st.caption(
        "AI-Integrated Airport Safety & Security Intelligence System · "
        "Phase 1 MVP · Dual-model person + PPE correlation"
    )

    fingerprint = weights_fingerprint(FINE_TUNED_WEIGHTS)
    base_model, ppe_model, is_trained = load_models(fingerprint)

    if not is_trained:
        st.error(
            "No trained PPE weights found at `models/best.pt`. "
            "Train the model (see `notebooks/`) and place `best.pt` in `models/`."
        )
        return

    with st.sidebar:
        st.header("Settings")
        person_conf = st.slider("Person confidence", 0.05, 0.9, 0.40, 0.05)
        ppe_conf = st.slider("PPE confidence", 0.05, 0.9, 0.40, 0.05)

        st.markdown("---")
        st.subheader("Conditions (optional)")
        station = st.text_input(
            "Airport ICAO code", value="", max_chars=4,
            placeholder="e.g. KCHS",
            help="Fetches current METAR to set weather-conditional PPE "
                 "requirements per docs/PPE_TAXONOMY.md. Leave blank to "
                 "use fail-safe defaults (see conditions.default_requirements).",
        ).strip().upper()

        weather = None
        if station:
            if is_valid_icao(station):
                weather = fetch_metar(station)
                if weather is None:
                    st.warning(f"Could not fetch METAR for {station}. Using fail-safe defaults.")
                elif weather.is_stale:
                    st.warning(f"METAR for {station} is stale. Using fail-safe defaults.")
                else:
                    st.success(f"{station}: {weather.wind_speed_kt} kt, {weather.temp_c}C")
                    st.caption(weather.raw_metar)
            else:
                st.warning("ICAO codes are 4 letters, e.g. KCHS.")

        requirements = determine_requirements(weather)
        with st.expander("Current PPE requirements"):
            for cls, reason in requirements.reasons.items():
                st.write(f"**{cls}**: {reason}")

        st.markdown("---")
        st.markdown(
            "**How this works**\n\n"
            "Every uploaded image is run through two models: a person "
            "detector and a fine-tuned PPE detector. The app checks "
            "whether each detected person overlaps with a vest detection "
            "- if not, that person is flagged as a violation."
        )
        st.markdown("---")
        with st.expander("Build diagnostics"):
            import torch
            import ultralytics

            st.write(
                {
                    "ultralytics": ultralytics.__version__,
                    "torch": torch.__version__,
                    "weights_sha256": fingerprint[:12],
                    "ppe_classes": ppe_model.names,
                }
            )
            st.caption(
                "Report `weights_sha256` when comparing local and deployed "
                "results - differing values mean different models."
            )

    uploaded = st.file_uploader("Upload a ramp / worksite photo", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.info("Upload an image to run detection.")
        return

    image = Image.open(uploaded).convert("RGB")

    # CRITICAL: ultralytics interprets a numpy array as BGR. Passing an RGB
    # array makes the model see colour-inverted pixels (hi-vis yellow becomes
    # blue), which silently destroys vest detection. Convert to BGR first.
    # Regression test: tests/test_channel_order.py
    image_bgr = np.array(image)[:, :, ::-1]

    with st.spinner("Running person detection + PPE detection..."):
        person_results = base_model(image_bgr, conf=person_conf, classes=[0])[0]
        ppe_results = ppe_model(image_bgr, conf=ppe_conf)[0]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("PPE Detections")
        st.image(annotated_rgb(ppe_results), use_container_width=True)

    summary = summarize_results(person_results, ppe_results, requirements=requirements)

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

    st.caption(
        "Scope note: `vest` is the validated Phase 1 class. `helmet` and "
        "`gloves` were trained on construction-domain imagery and do not yet "
        "transfer reliably to airport ramp operations - they are reported for "
        "transparency but are not used in the compliance decision."
    )

    with st.expander("Raw detections (all classes, with confidence)"):
        if ppe_results.boxes is not None and len(ppe_results.boxes) > 0:
            rows = [
                {
                    "class": ppe_results.names[int(b.cls[0])],
                    "confidence": round(float(b.conf[0]), 3),
                    "box_xyxy": [round(v, 1) for v in b.xyxy[0].tolist()],
                }
                for b in ppe_results.boxes
            ]
            st.dataframe(rows, use_container_width=True)
        else:
            st.write("No PPE items detected above the current threshold.")

    if summary.people:
        with st.expander("Per-person compliance detail"):
            for i, person in enumerate(summary.people, 1):
                items = "  ".join(
                    f"{cls}={status.value}" for cls, status in person.status.items()
                )
                st.write(f"Person {i} (conf {person.confidence}): {items}")
            st.caption(
                "INDETERMINATE = not observed, but this class is not yet "
                "reliable enough (or not currently required) to count as a "
                "violation. Only VIOLATION affects the compliance rate."
            )


if __name__ == "__main__":
    main()
