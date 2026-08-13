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

st.set_page_config(page_title="ASSIS - PPE Compliance", page_icon="🛬", layout="wide")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --assis-ink: #0F1720;
    --assis-panel: #17212B;
    --assis-line: #2A3742;
    --assis-text: #E6EBF0;
    --assis-muted: #8B99A6;
    --assis-amber: #E8A33D;
    --assis-green: #4CAF7D;
    --assis-red: #E0604E;
}

html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
code, .stCodeBlock, [data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; }

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

.stApp { background: var(--assis-ink); color: var(--assis-text); }

section[data-testid="stSidebar"] {
    background: var(--assis-panel);
    border-right: 1px solid var(--assis-line);
}

.assis-header {
    display: flex; align-items: center; gap: 14px;
    padding: 4px 0 20px 0; margin-bottom: 8px;
    border-bottom: 1px solid var(--assis-line);
}
.assis-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 44px; height: 44px; border-radius: 8px;
    background: linear-gradient(135deg, var(--assis-amber), #C97F1E);
    font-size: 22px; flex-shrink: 0;
}
.assis-title { font-size: 22px; font-weight: 700; letter-spacing: 0.2px; color: var(--assis-text); line-height: 1.2; }
.assis-subtitle { font-size: 13px; color: var(--assis-muted); margin-top: 2px; }
.assis-eyebrow {
    font-size: 11px; font-weight: 600; letter-spacing: 1.2px; text-transform: uppercase;
    color: var(--assis-amber); margin-bottom: 6px;
}

.assis-card {
    background: var(--assis-panel); border: 1px solid var(--assis-line);
    border-radius: 10px; padding: 18px 20px; height: 100%;
}
.assis-card-label {
    font-size: 11px; font-weight: 600; letter-spacing: 0.8px; text-transform: uppercase;
    color: var(--assis-muted); margin-bottom: 6px;
}
.assis-card-value { font-family: 'JetBrains Mono', monospace; font-size: 30px; font-weight: 500; color: var(--assis-text); }
.assis-card-value.amber { color: var(--assis-amber); }
.assis-card-value.green { color: var(--assis-green); }
.assis-card-value.red { color: var(--assis-red); }

.assis-status {
    border-radius: 10px; padding: 14px 18px; font-weight: 600; font-size: 14px;
    display: flex; align-items: center; gap: 10px; letter-spacing: 0.2px;
}
.assis-status.ok { background: rgba(76,175,125,0.12); border: 1px solid rgba(76,175,125,0.35); color: var(--assis-green); }
.assis-status.violation { background: rgba(224,96,78,0.12); border: 1px solid rgba(224,96,78,0.35); color: var(--assis-red); }
.assis-status.neutral { background: rgba(139,153,166,0.10); border: 1px solid var(--assis-line); color: var(--assis-muted); }

.assis-scope-note {
    font-size: 12.5px; color: var(--assis-muted); line-height: 1.5;
    border-left: 2px solid var(--assis-line); padding-left: 12px; margin-top: 14px;
}
.assis-scope-note code {
    background: var(--assis-line); padding: 1px 5px; border-radius: 4px; font-size: 11.5px;
}

[data-testid="stImage"] img { border-radius: 8px; border: 1px solid var(--assis-line); }
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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
    st.markdown(
        """
        <div class="assis-header">
            <div class="assis-badge">🛬</div>
            <div>
                <div class="assis-title">ASSIS — PPE Compliance Detection</div>
                <div class="assis-subtitle">AI-Integrated Airport Safety &amp; Security Intelligence System · Phase 1 · Vest-gated compliance with weather-conditional policy</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
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
        st.markdown('<div class="assis-eyebrow">Detection thresholds</div>', unsafe_allow_html=True)
        person_conf = st.slider("Person confidence", 0.05, 0.9, 0.50, 0.05)
        if person_conf < 0.40:
            st.caption(
                "⚠️ Below ~0.40, the person detector can pick up false "
                "positives (background shapes, vehicle structure) that have "
                "no PPE overlapping them, which are then counted as "
                "violations. Use 0.5+ for operational results; lower only "
                "to stress-test."
            )
        ppe_conf = st.slider("PPE confidence", 0.05, 0.9, 0.40, 0.05)

        st.markdown("---")
        st.markdown('<div class="assis-eyebrow">Conditions</div>', unsafe_allow_html=True)
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
                    st.success(f"{station} · {weather.wind_speed_kt} kt · {weather.temp_c}°C")
                    st.caption(weather.raw_metar)
            else:
                st.warning("ICAO codes are 4 letters, e.g. KCHS.")

        requirements = determine_requirements(weather)
        with st.expander("Current PPE requirements"):
            for cls, reason in requirements.reasons.items():
                st.write(f"**{cls}**: {reason}")

        st.markdown("---")
        st.markdown('<div class="assis-eyebrow">How this works</div>', unsafe_allow_html=True)
        st.caption(
            "Every uploaded image runs through two models: a person "
            "detector and a fine-tuned PPE detector. A person is compliant "
            "if a vest detection overlaps their bounding box; if not, "
            "they're flagged as a violation."
        )
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
        st.markdown(
            '<div class="assis-status neutral">📷 Upload an image to run detection.</div>',
            unsafe_allow_html=True,
        )
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
        st.markdown('<div class="assis-eyebrow">Original</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
    with col2:
        st.markdown('<div class="assis-eyebrow">PPE detections</div>', unsafe_allow_html=True)
        st.image(annotated_rgb(ppe_results), use_container_width=True)

    summary = summarize_results(person_results, ppe_results, requirements=requirements)

    st.markdown('<div class="assis-eyebrow" style="margin-top:28px;">Compliance summary</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    violation_class = "red" if summary.violations else ""
    compliant_class = "green" if summary.total_people and not summary.violations else ""
    cards = [
        (c1, "Personnel detected", summary.total_people, ""),
        (c2, "Vest compliant", summary.vest_compliant, compliant_class),
        (c3, "Violations", summary.violations, violation_class),
        (c4, "Compliance rate", f"{summary.compliance_rate}%", compliant_class or violation_class),
    ]
    for col, label, value, cls in cards:
        col.markdown(
            f"""<div class="assis-card">
                    <div class="assis-card-label">{label}</div>
                    <div class="assis-card-value {cls}">{value}</div>
                </div>""",
            unsafe_allow_html=True,
        )

    st.write("")
    if summary.violations > 0:
        st.markdown(f'<div class="assis-status violation">⛔ {summary.status}</div>', unsafe_allow_html=True)
    elif summary.total_people > 0:
        st.markdown(f'<div class="assis-status ok">✅ {summary.status}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assis-status neutral">— {summary.status}</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="assis-scope-note">Scope note: <code>vest</code> is the '
        "validated Phase 1 class. <code>helmet</code> and <code>gloves</code> "
        "were trained on construction-domain imagery and do not yet transfer "
        "reliably to airport ramp operations — they are reported for "
        "transparency but are not used in the compliance decision.</div>",
        unsafe_allow_html=True,
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
