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

st.set_page_config(page_title="ASSIS — RampGuard", page_icon="✈", layout="wide")

if "theme" not in st.session_state:
    st.session_state.theme = "light"

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
def render_css(theme: str) -> str:
    """Light theme by default (BLADE-inspired: white, generous whitespace,
    bold sans headline, thin gray card borders, one blue accent), with a
    dark variant toggled from the sidebar. See docs/UI_UX_BLUEPRINT.md."""
    if theme == "dark":
        tokens = {
            "bg": "#0B132B", "panel": "#1C2541", "line": "#2E3A63",
            "text": "#F4F6F9", "dim": "#8D99AE", "accent": "#2F6FED",
            "alert": "#EF233C", "caution": "#FFB703", "ok": "#38B000",
            "card_shadow": "none",
        }
    else:
        tokens = {
            "bg": "#FFFFFF", "panel": "#FFFFFF", "line": "#E4E7EC",
            "text": "#101828", "dim": "#667085", "accent": "#2F6FED",
            "alert": "#D92D20", "caution": "#B54708", "ok": "#12805C",
            "card_shadow": "0 1px 2px rgba(16,24,40,0.05)",
        }

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {{
    --bg: {tokens['bg']}; --panel: {tokens['panel']}; --line: {tokens['line']};
    --text: {tokens['text']}; --dim: {tokens['dim']}; --accent: {tokens['accent']};
    --alert: {tokens['alert']}; --caution: {tokens['caution']}; --ok: {tokens['ok']};
    --shadow: {tokens['card_shadow']};
}}

html, body, [class*="css"] {{ font-family: 'Inter', -apple-system, sans-serif; }}
.mono, code, [data-testid="stDataFrame"] {{ font-family: 'IBM Plex Mono', monospace; }}

#MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; height: 0; }}

.stApp {{ background: var(--bg); color: var(--text); font-size: 16px; }}

section[data-testid="stSidebar"] {{
    background: var(--bg); border-right: 1px solid var(--line);
}}
section[data-testid="stSidebar"] p {{ color: var(--dim) !important; font-size: 14px; }}

button, .stButton button {{ min-height: 44px; }}

/* Hero header - BLADE-style: bold oversized headline, muted subhead, plenty of air */
.hero {{
    padding: 28px 0 32px 0; border-bottom: 1px solid var(--line); margin-bottom: 20px;
}}
.hero-eyebrow {{
    font-size: 12px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase;
    color: var(--accent); margin-bottom: 10px;
}}
.hero-title {{
    font-size: 40px; font-weight: 800; letter-spacing: -0.5px; color: var(--text);
    line-height: 1.05; margin-bottom: 10px;
}}
.hero-subtitle {{ font-size: 15px; color: var(--dim); max-width: 620px; line-height: 1.5; }}

.section-tag {{
    display: inline-flex; align-items: center; font-size: 12px; font-weight: 600;
    letter-spacing: 0.5px; text-transform: uppercase; color: var(--dim);
    margin-bottom: 10px; gap: 6px;
}}
.section-tag::before {{ content: ''; width: 6px; height: 6px; border-radius: 50%; background: var(--accent); }}

[data-testid="stFileUploader"] section {{
    background: var(--panel);
    border: 1.5px dashed var(--line) !important;
    border-radius: 12px; box-shadow: var(--shadow);
    transition: border-color 0.15s ease;
}}
[data-testid="stFileUploader"] section:hover {{ border-color: var(--accent) !important; }}

.metar-panel {{
    background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
    padding: 18px 20px; box-shadow: var(--shadow); position: sticky; top: 12px;
}}
.metar-station-row {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 12px; }}
.metar-icao {{ font-family: 'IBM Plex Mono', monospace; font-size: 22px; font-weight: 700; color: var(--text); }}
.metar-flightcat {{
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 700; letter-spacing: 0.6px;
    padding: 3px 9px; border-radius: 5px; background: rgba(18,128,92,0.12); color: var(--ok);
}}
.metar-flightcat.ifr {{ background: rgba(217,45,32,0.12); color: var(--alert); }}
.metar-grid {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 10px 16px;
    border-top: 1px solid var(--line); border-bottom: 1px solid var(--line);
    padding: 12px 0; margin-bottom: 10px;
}}
.metar-field-label {{ font-size: 12px; color: var(--dim); }}
.metar-field-value {{ font-family: 'IBM Plex Mono', monospace; font-size: 18px; font-weight: 500; color: var(--text); margin-top: 2px; }}
.metar-raw {{ font-family: 'IBM Plex Mono', monospace; font-size: 11.5px; color: var(--dim); line-height: 1.5; }}
.metar-empty {{ font-size: 13px; color: var(--dim); }}
.metar-advisory {{
    display: flex; gap: 8px; align-items: flex-start; font-size: 12.5px;
    background: rgba(181,71,8,0.08); border: 1px solid rgba(181,71,8,0.25); color: var(--caution);
    border-radius: 8px; padding: 8px 10px; margin-top: 10px;
}}

.readout {{
    border: 1px solid var(--line); border-radius: 12px; padding: 18px; height: 100%;
    background: var(--panel); box-shadow: var(--shadow);
}}
.readout-label {{ font-size: 12px; color: var(--dim); }}
.readout-value {{ font-family: 'IBM Plex Mono', monospace; font-size: 28px; font-weight: 600; margin-top: 4px; color: var(--text); }}
.readout-value.ok {{ color: var(--ok); }}
.readout-value.alert {{ color: var(--alert); }}

.verdict {{
    border-radius: 10px; padding: 14px 18px; font-size: 15px; font-weight: 700;
    display: flex; gap: 10px; align-items: center; background: var(--panel);
    border: 1px solid var(--line); border-left: 4px solid var(--ok); color: var(--ok);
}}
.verdict.alert {{ border-left-color: var(--alert); color: var(--alert); }}
.verdict.dim {{ border-left-color: var(--line); color: var(--dim); }}

.scope-note {{
    font-size: 12.5px; color: var(--dim); line-height: 1.6;
    border-top: 1px solid var(--line); padding-top: 10px; margin-top: 14px;
}}
.scope-note code {{ font-family: 'IBM Plex Mono', monospace; background: var(--line); color: var(--text); padding: 1px 5px; border-radius: 3px; }}

[data-testid="stImage"] img {{ border-radius: 12px; border: 1px solid var(--line); }}

div[data-testid="stHorizontalBlock"]:has(.metar-panel-marker) {{ align-items: flex-start; }}
@media (max-width: 768px) {{
    div[data-testid="stHorizontalBlock"]:has(.metar-panel-marker) {{ flex-direction: column !important; }}
    div[data-testid="stHorizontalBlock"]:has(.metar-panel-marker) > div {{ width: 100% !important; flex: 1 1 100% !important; }}
    .metar-panel {{ position: static; margin-top: 16px; }}
    .hero-title {{ font-size: 30px; }}
}}
</style>
"""


st.markdown(render_css(st.session_state.theme), unsafe_allow_html=True)


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
    """Rough VFR/IFR classification from wind, purely for the briefing-strip
    badge - not a substitute for an actual flight-category calculation
    (which also needs ceiling and visibility)."""
    if weather is None:
        return "—", "dim"
    if weather.wind_speed_kt is not None and weather.wind_speed_kt >= 30:
        return "IFR", "ifr"
    return "VFR", "ok"


def main() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">ASSIS · Phase 1</div>
            <div class="hero-title">RampGuard</div>
            <div class="hero-subtitle">Automated PPE compliance for airport ramp operations.
            Upload a photo to check hi-vis vest compliance, weighed against live weather
            conditions at the airport.</div>
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

    feed_col, metar_col = st.columns([7, 3])

    with metar_col:
        st.markdown('<div class="metar-panel-marker"></div>', unsafe_allow_html=True)
        st.markdown('<span class="section-tag">Weather briefing</span>', unsafe_allow_html=True)
        station = st.text_input(
            "Airport ICAO code", value="", max_chars=4, placeholder="KCHS",
            label_visibility="collapsed",
        ).strip().upper()
        st.caption("4-letter ICAO code (e.g. KCHS). Sets weather-conditional PPE requirements. Blank = fail-safe defaults.")

        weather = None
        metar_html = ""
        if station and not is_valid_icao(station):
            metar_html = '<div class="metar-panel"><span class="metar-empty">⚠ ICAO codes are 4 letters, e.g. KCHS.</span></div>'
        elif station:
            weather = fetch_metar(station)
            if weather is None:
                metar_html = f'<div class="metar-panel"><span class="metar-empty">⚠ No data for {station}. Fail-safe requirements applied.</span></div>'
            elif weather.is_stale:
                metar_html = f'<div class="metar-panel"><span class="metar-empty">⚠ Stale observation for {station}. Fail-safe requirements applied.</span></div>'
            else:
                cat, cat_cls = flight_category(weather)
                metar_html = f"""<div class="metar-panel">
                    <div class="metar-station-row">
                        <span class="metar-icao">{station}</span>
                        <span class="metar-flightcat {cat_cls}">{cat}</span>
                    </div>
                    <div class="metar-grid">
                        <div><div class="metar-field-label">WIND</div><div class="metar-field-value">{weather.wind_speed_kt} kt</div></div>
                        <div><div class="metar-field-label">TEMP</div><div class="metar-field-value">{weather.temp_c}°C</div></div>
                    </div>
                    <div class="metar-raw">{weather.raw_metar}</div>
                </div>"""
        else:
            metar_html = '<div class="metar-panel"><span class="metar-empty">No station set — fail-safe PPE requirements (all gating classes required).</span></div>'

        st.markdown(metar_html, unsafe_allow_html=True)

        requirements = determine_requirements(weather)
        conditional_notes = [
            f"{cls}: {reason}" for cls, reason in requirements.reasons.items()
            if "Required:" in reason
        ]
        if conditional_notes:
            st.markdown(
                '<div class="metar-advisory">⚠ ' + "<br>".join(conditional_notes) + '</div>',
                unsafe_allow_html=True,
            )
        with st.expander("All PPE requirements"):
            for cls, reason in requirements.reasons.items():
                st.markdown(f"**{cls}** — {reason}")

    with feed_col:
        st.markdown('<span class="section-tag">Imagery</span>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload a ramp / worksite photo", type=["jpg", "jpeg", "png"])

        if uploaded is None:
            st.markdown('<div class="verdict dim">— awaiting image —</div>', unsafe_allow_html=True)
            uploaded_none = True
        else:
            uploaded_none = False

    if uploaded_none:
        return

    with st.sidebar:
        st.markdown('<span class="section-tag dim">Appearance</span>', unsafe_allow_html=True)
        theme_choice = st.radio(
            "Theme", ["Light", "Dark"],
            index=0 if st.session_state.theme == "light" else 1,
            horizontal=True, label_visibility="collapsed",
        )
        new_theme = theme_choice.lower()
        if new_theme != st.session_state.theme:
            st.session_state.theme = new_theme
            st.rerun()

        st.markdown("---")
        st.markdown('<span class="section-tag dim">Thresholds</span>', unsafe_allow_html=True)
        person_conf = st.slider("Person confidence", 0.05, 0.9, 0.50, 0.05)
        if person_conf < 0.40:
            st.caption(
                "⚠ Below ~0.40 the person detector can pick up false "
                "positives (background shapes, vehicle structure) with no "
                "PPE overlapping them, counted as violations. Use 0.5+ "
                "operationally; lower only to stress-test."
            )
        ppe_conf = st.slider("PPE confidence", 0.05, 0.9, 0.60, 0.05)
        st.caption(
            "Default raised to 0.60 after observing a false positive at "
            "0.44 (a plain shirt misread as a vest in a non-ramp scene). "
            "Confirmed genuine vest detections in testing cluster at "
            "0.88–0.92 — there is a real gap between true detections and "
            "noise around 0.40–0.55. Treat detections in that band as "
            "unverified, not confirmed."
        )

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

    with feed_col:
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
        v_cls = "alert" if summary.violations else ""
        ok_cls = "ok" if summary.total_people and not summary.violations else ""
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
            st.markdown(f'<div class="verdict alert">⛔ &nbsp; {summary.status}</div>', unsafe_allow_html=True)
        elif summary.total_people > 0:
            st.markdown(f'<div class="verdict ok">✓ &nbsp; {summary.status}</div>', unsafe_allow_html=True)
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
                        "flag": "⚠ unverified — below 0.70" if float(b.conf[0]) < 0.70 else "",
                        "box_xyxy": [round(v, 1) for v in b.xyxy[0].tolist()],
                    }
                    for b in ppe_results.boxes
                ]
                st.dataframe(rows, use_container_width=True)
                st.caption(
                    "Confirmed genuine vest detections in testing scored "
                    "0.88–0.92. Anything below ~0.70 has not been validated "
                    "as reliable and should be treated as unverified."
                )
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
