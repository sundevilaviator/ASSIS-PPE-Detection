"""
ASSIS PPE Detection - operational demo.

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

st.set_page_config(page_title="ASSIS PPE // Briefing", page_icon="✈", layout="wide")

# ---------------------------------------------------------------------------
# Design system.
#
# Grounded in real aviation weather-briefing terminals and ATC scopes, which
# are monochrome phosphor displays (green or amber on black), monospace,
# with color reserved for ICAO flight-category coding:
#   VFR = green, MVFR = blue, IFR = red, LIFR = magenta.
# That coding is reused directly for compliance state, since both are
# "is it safe to proceed" signals in the same operational vernacular.
# This is a deliberate departure from a generic dark dashboard: monospace is
# the PRIMARY typeface (not just a data accent), rules are drawn as dashed
# hairlines evoking chart lines, and the layout is organized as a briefing
# strip rather than a card grid.
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
    --scope-bg: #0A0D0A;
    --scope-panel: #10140F;
    --scope-line: #23301F;
    --scope-text: #C9D6C4;
    --scope-dim: #5C6B57;
    --vfr: #3FBF6B;
    --mvfr: #4A90D9;
    --ifr: #D9524A;
    --lifr: #C24AD9;
    --caution: #D9A23F;
}

html, body, [class*="css"] { font-family: 'IBM Plex Mono', monospace; }
h1, h2, h3, p, span, div { font-family: 'IBM Plex Mono', monospace; }
.assis-prose { font-family: 'IBM Plex Sans', sans-serif; }

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

.stApp { background: var(--scope-bg); color: var(--scope-text); }
.stApp, .stApp * { scrollbar-color: var(--scope-line) var(--scope-bg); }

section[data-testid="stSidebar"] {
    background: var(--scope-panel);
    border-right: 1px dashed var(--scope-line);
}
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p { color: var(--scope-dim) !important; font-size: 12.5px; }

/* Top strip: callsign-style identity bar */
.strip-header {
    display: flex; justify-content: space-between; align-items: baseline;
    padding: 10px 0 12px 0; border-bottom: 1px dashed var(--scope-line);
    margin-bottom: 4px;
}
.strip-callsign { font-size: 19px; font-weight: 700; letter-spacing: 3px; color: var(--vfr); }
.strip-meta { font-size: 11px; color: var(--scope-dim); letter-spacing: 0.5px; }

.section-tag {
    display: inline-block; font-size: 10.5px; font-weight: 600; letter-spacing: 2px;
    text-transform: uppercase; color: var(--scope-bg); background: var(--vfr);
    padding: 2px 8px; margin-bottom: 10px;
}
.section-tag.blue { background: var(--mvfr); }
.section-tag.dim { background: var(--scope-dim); }

/* METAR briefing strip - the signature element */
.metar-strip {
    background: var(--scope-panel); border: 1px dashed var(--scope-line);
    padding: 14px 18px; margin-bottom: 4px;
}
.metar-station-row { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; }
.metar-icao { font-size: 24px; font-weight: 700; letter-spacing: 2px; }
.metar-icao.vfr { color: var(--vfr); }
.metar-icao.pending { color: var(--scope-dim); }
.metar-flightcat {
    font-size: 10.5px; font-weight: 700; letter-spacing: 1.5px; padding: 2px 7px;
    border: 1px solid currentColor;
}
.metar-flightcat.vfr { color: var(--vfr); }
.metar-readout {
    font-size: 12.5px; color: var(--scope-text); margin-top: 8px;
    letter-spacing: 0.5px; word-spacing: 4px;
}
.metar-raw {
    font-size: 11px; color: var(--scope-dim); margin-top: 6px; letter-spacing: 0.3px;
}
.metar-empty { font-size: 12px; color: var(--scope-dim); }

/* Compliance readout cards */
.readout {
    border: 1px dashed var(--scope-line); padding: 14px 16px; height: 100%;
}
.readout-label { font-size: 10px; letter-spacing: 1.5px; text-transform: uppercase; color: var(--scope-dim); }
.readout-value { font-size: 28px; font-weight: 600; margin-top: 4px; }
.readout-value.vfr { color: var(--vfr); }
.readout-value.ifr { color: var(--ifr); }

.verdict {
    border-left: 3px solid var(--vfr); padding: 12px 16px; font-size: 13.5px;
    font-weight: 600; letter-spacing: 0.5px; display: flex; gap: 10px; align-items: center;
}
.verdict.ifr { border-left-color: var(--ifr); color: var(--ifr); }
.verdict.vfr { color: var(--vfr); }
.verdict.dim { border-left-color: var(--scope-line); color: var(--scope-dim); }

.scope-note {
    font-size: 11.5px; color: var(--scope-dim); line-height: 1.6;
    border-top: 1px dashed var(--scope-line); padding-top: 10px; margin-top: 14px;
}
.scope-note code { background: var(--scope-line); color: var(--scope-text); padding: 1px 5px; }

[data-testid="stImage"] img { border: 1px dashed var(--scope-line); }
[data-testid="stFileUploader"] section { background: var(--scope-panel); border: 1px dashed var(--scope-line) !important; }
.stSlider { padding-top: 4px; }
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


def flight_category(weather) -> tuple[str, str]:
    """Rough VFR/MVFR/IFR/LIFR classification from wind, purely for the
    briefing-strip badge - not a substitute for an actual flight-category
    calculation (which also needs ceiling and visibility)."""
    if weather is None:
        return "—", "dim"
    if weather.wind_speed_kt is not None and weather.wind_speed_kt >= 30:
        return "IFR", "ifr"
    return "VFR", "vfr"


def main() -> None:
    st.markdown(
        """
        <div class="strip-header">
            <div class="strip-callsign">ASSIS // PPE BRIEFING</div>
            <div class="strip-meta">PHASE 1 · VEST-GATED COMPLIANCE · METAR-CONDITIONAL POLICY</div>
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

    # -----------------------------------------------------------------
    # METAR briefing strip - moved to the main canvas, not the sidebar,
    # because it's a primary input to the compliance decision and was
    # easy to miss tucked in the sidebar.
    # -----------------------------------------------------------------
    st.markdown('<span class="section-tag">Weather briefing</span>', unsafe_allow_html=True)
    bcol1, bcol2 = st.columns([1, 3])
    with bcol1:
        station = st.text_input(
            "Airport ICAO code", value="", max_chars=4, placeholder="KCHS",
            label_visibility="collapsed",
        ).strip().upper()
    with bcol2:
        st.caption("Enter a 4-letter ICAO code (e.g. KCHS) to pull live METAR and set weather-conditional PPE requirements. Leave blank for fail-safe defaults.")

    weather = None
    if station:
        if is_valid_icao(station):
            weather = fetch_metar(station)
        else:
            st.markdown('<div class="metar-strip"><span class="metar-empty">⚠ ICAO codes are 4 letters, e.g. KCHS.</span></div>', unsafe_allow_html=True)

    if station and is_valid_icao(station):
        if weather is None:
            st.markdown(f'<div class="metar-strip"><span class="metar-empty">⚠ NO DATA — could not reach METAR for {station}. Fail-safe requirements applied.</span></div>', unsafe_allow_html=True)
        elif weather.is_stale:
            st.markdown(f'<div class="metar-strip"><span class="metar-empty">⚠ STALE — last observation for {station} exceeds 90 min. Fail-safe requirements applied.</span></div>', unsafe_allow_html=True)
        else:
            cat, cat_cls = flight_category(weather)
            st.markdown(
                f"""<div class="metar-strip">
                        <div class="metar-station-row">
                            <span class="metar-icao vfr">{station}</span>
                            <span class="metar-flightcat {cat_cls}">{cat}</span>
                            <span class="metar-readout">WIND {weather.wind_speed_kt} KT &nbsp;·&nbsp; TEMP {weather.temp_c}°C</span>
                        </div>
                        <div class="metar-raw">{weather.raw_metar}</div>
                    </div>""",
                unsafe_allow_html=True,
            )
    elif not station:
        st.markdown('<div class="metar-strip"><span class="metar-empty">— no station set — using fail-safe PPE requirements (all gating classes required)</span></div>', unsafe_allow_html=True)

    requirements = determine_requirements(weather)
    with st.expander("PPE requirements under current conditions"):
        for cls, reason in requirements.reasons.items():
            st.markdown(f"**{cls}** — {reason}")

    st.write("")

    with st.sidebar:
        st.markdown('<span class="section-tag dim">Thresholds</span>', unsafe_allow_html=True)
        person_conf = st.slider("Person confidence", 0.05, 0.9, 0.50, 0.05)
        if person_conf < 0.40:
            st.caption(
                "⚠ Below ~0.40 the person detector can pick up false "
                "positives (background shapes, vehicle structure) with no "
                "PPE overlapping them, counted as violations. Use 0.5+ "
                "operationally; lower only to stress-test."
            )
        ppe_conf = st.slider("PPE confidence", 0.05, 0.9, 0.40, 0.05)

        st.markdown("---")
        st.markdown('<span class="section-tag dim">How this works</span>', unsafe_allow_html=True)
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

    st.markdown('<span class="section-tag">Imagery</span>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a ramp / worksite photo", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.markdown('<div class="verdict dim">— awaiting image —</div>', unsafe_allow_html=True)
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
        st.markdown('<span class="section-tag dim">Original</span>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
    with col2:
        st.markdown('<span class="section-tag dim">Detections</span>', unsafe_allow_html=True)
        st.image(annotated_rgb(ppe_results), use_container_width=True)

    summary = summarize_results(person_results, ppe_results, requirements=requirements)

    st.write("")
    st.markdown('<span class="section-tag">Compliance readout</span>', unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)
    v_cls = "ifr" if summary.violations else ""
    ok_cls = "vfr" if summary.total_people and not summary.violations else ""
    for col, label, value, cls in [
        (r1, "Personnel", summary.total_people, ""),
        (r2, "Vest compliant", summary.vest_compliant, ok_cls),
        (r3, "Violations", summary.violations, v_cls),
        (r4, "Compliance rate", f"{summary.compliance_rate}%", ok_cls or v_cls),
    ]:
        col.markdown(
            f'<div class="readout"><div class="readout-label">{label}</div>'
            f'<div class="readout-value {cls}">{value}</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    if summary.violations > 0:
        st.markdown(f'<div class="verdict ifr">⛔ &nbsp; {summary.status}</div>', unsafe_allow_html=True)
    elif summary.total_people > 0:
        st.markdown(f'<div class="verdict vfr">✓ &nbsp; {summary.status}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="verdict dim">— &nbsp; {summary.status}</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="scope-note">SCOPE: <code>vest</code> is the validated '
        "Phase 1 class. <code>helmet</code> and <code>gloves</code> were "
        "trained on construction-domain imagery and do not yet transfer "
        "reliably to airport ramp operations — reported for transparency, "
        "not used in the compliance decision.</div>",
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
