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

.feature-grid {{
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;
    margin: 24px 0 8px 0;
}}
.feature-card {{
    background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
    padding: 20px; box-shadow: var(--shadow);
}}
.feature-icon {{ font-size: 22px; margin-bottom: 8px; }}
.feature-title {{ font-size: 14.5px; font-weight: 700; color: var(--text); margin-bottom: 6px; }}
.feature-body {{ font-size: 12.5px; color: var(--dim); line-height: 1.5; }}
@media (max-width: 768px) {{
    .feature-grid {{ grid-template-columns: 1fr; }}
}}

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


def _estimate_unique_people(frame_results: list) -> int:
    """Estimate a deduplicated person count across sampled video frames using
    greedy centroid matching between consecutive sampled frames.

    This is a heuristic, not true re-identification. It has a real, specific
    limitation worth stating plainly: frames are sampled every ~2 seconds
    (SAMPLE_EVERY_N_SECONDS in run_video_pipeline), and a person can move far
    enough in 2 seconds that this matching under-merges (counts one real
    person as multiple) - especially at typical walking speed across a wide
    apron shot. It will not over-merge two different people into one unless
    they happen to occupy nearly the same position in consecutive sampled
    frames, which is uncommon. Treat the result as a better estimate than
    raw per-frame summation, not as an exact count. A reliable solution
    needs either continuous (not sparse) frame tracking with a real
    tracker (e.g. ByteTrack/DeepSORT, both already available via
    ultralytics' built-in `model.track()`) or re-identification embeddings -
    both are real future work, not implemented here.
    """
    MATCH_DISTANCE_FRACTION = 0.08  # of frame diagonal - matched empirically, not tuned on real data

    next_id = 0
    tracked_centroids: dict[int, tuple[float, float]] = {}

    for frame in frame_results:
        h, w = frame["frame_rgb"].shape[:2]
        diagonal = (h ** 2 + w ** 2) ** 0.5
        max_dist = diagonal * MATCH_DISTANCE_FRACTION

        current_centroids = []
        for person in frame["summary"].people:
            x1, y1, x2, y2 = person.person_bbox
            current_centroids.append(((x1 + x2) / 2, (y1 + y2) / 2))

        matched_ids = set()
        for cx, cy in current_centroids:
            best_id, best_dist = None, max_dist
            for tid, (tx, ty) in tracked_centroids.items():
                if tid in matched_ids:
                    continue
                dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
                if dist < best_dist:
                    best_id, best_dist = tid, dist
            if best_id is not None:
                matched_ids.add(best_id)
                tracked_centroids[best_id] = (cx, cy)
            else:
                tracked_centroids[next_id] = (cx, cy)
                matched_ids.add(next_id)
                next_id += 1

    return next_id


def run_video_pipeline(uploaded_video, base_model, ppe_model, person_conf, ppe_conf, requirements) -> None:
    """Sample frames from an uploaded video and run the same detection
    pipeline used for single images on each sampled frame.

    This is frame-by-frame batch processing of an uploaded file, not live
    video streaming. A real CCTV/RTSP ingestion pipeline would need a
    different architecture (continuous capture, not a file upload) - see
    docs/FOD_PHASE2_PLAN.md Section 4.1 for what that would require.
    """
    import cv2
    import tempfile

    SAMPLE_EVERY_N_SECONDS = 2.0
    MAX_FRAMES = 30  # cap so a long video doesn't stall the UI or the free tier

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix) as tmp:
        tmp.write(uploaded_video.read())
        video_path = tmp.name

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(fps * SAMPLE_EVERY_N_SECONDS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps if fps else 0.0

    st.caption(
        f"Video: {duration_s:.1f}s at ~{fps:.0f} fps. Sampling one frame every "
        f"{SAMPLE_EVERY_N_SECONDS:.0f}s (up to {MAX_FRAMES} frames)."
    )

    frame_results = []
    frame_idx = 0
    progress = st.progress(0, text="Sampling and running detection on frames...")

    while cap.isOpened() and len(frame_results) < MAX_FRAMES:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            timestamp_s = frame_idx / fps if fps else 0.0
            person_results = base_model(frame_bgr, conf=person_conf, classes=[0], verbose=False)[0]
            ppe_results = ppe_model(frame_bgr, conf=ppe_conf, verbose=False)[0]
            summary = summarize_results(person_results, ppe_results, requirements=requirements, frame_height=frame_bgr.shape[0])
            frame_rgb = frame_bgr[:, :, ::-1]
            annotated = annotated_rgb(ppe_results)
            frame_results.append(
                {
                    "timestamp_s": timestamp_s,
                    "frame_rgb": frame_rgb,
                    "annotated": annotated,
                    "summary": summary,
                }
            )
            if total_frames:
                progress.progress(min(1.0, frame_idx / total_frames), text=f"Frame at {timestamp_s:.1f}s...")
        frame_idx += 1

    cap.release()
    progress.empty()
    Path(video_path).unlink(missing_ok=True)

    if not frame_results:
        st.markdown('<div class="verdict dim">— no frames could be read from this video —</div>', unsafe_allow_html=True)
        return

    unique_person_count = _estimate_unique_people(frame_results)

    total_violations = sum(r["summary"].violations for r in frame_results)
    total_people = sum(r["summary"].total_people for r in frame_results)
    total_excluded_small = sum(r["summary"].excluded_small for r in frame_results)

    st.write("")
    st.markdown('<span class="section-tag">Video summary</span>', unsafe_allow_html=True)
    v1, v2, v3, v4 = st.columns(4)
    for col, label, value in [
        (v1, "Frames sampled", len(frame_results)),
        (v2, "Unique people (estimated)", unique_person_count),
        (v3, "Raw person-detections", total_people),
        (v4, "Total violations", total_violations),
    ]:
        cls = "alert" if label == "Total violations" and value else ""
        col.markdown(
            f'<div class="readout"><div class="readout-label">{label}</div>'
            f'<div class="readout-value {cls}">{value}</div></div>',
            unsafe_allow_html=True,
        )

    st.caption(
        "**Unique people (estimated)** is deduplicated across sampled frames "
        "using position-based matching between consecutive samples — a "
        "heuristic estimate, not exact tracking. **Raw person-detections** "
        "is the un-deduplicated sum shown before (one person visible in 5 "
        "sampled frames counts 5 times) — kept for comparison. Given ~2s "
        "gaps between sampled frames, a fast-moving person can still be "
        "counted more than once; treat 'Unique people' as an improved "
        "estimate, not an exact count."
    )

    if total_excluded_small:
        st.caption(
            f"ℹ {total_excluded_small} small/distant person detections across all "
            "sampled frames were excluded from compliance scoring (background "
            "people, passengers on jet bridges, etc. — not close enough in "
            "frame to reliably assess PPE). This is a partial mitigation for "
            "crowded/wide-shot scenes, not a full fix — see the note in "
            "src/utils.py (MIN_PERSON_HEIGHT_FRACTION) for what a complete "
            "solution would need."
        )

    st.caption(
        "Violation counts are still summed per-frame, not deduplicated by "
        "person — in crowded scenes (many people, wide shots), treat "
        "violation counts as directional signal, not a precise count. Role "
        "(ramp worker vs. passenger/bystander) cannot currently be "
        "distinguished from a bounding box alone."
    )

    st.write("")
    with st.expander(f"Per-frame detail ({len(frame_results)} sampled frames)", expanded=False):
        for r in frame_results:
            fc1, fc2 = st.columns(2)
            with fc1:
                st.image(r["frame_rgb"], caption=f"t={r['timestamp_s']:.1f}s — original", use_container_width=True)
            with fc2:
                st.image(r["annotated"], caption=f"t={r['timestamp_s']:.1f}s — detections", use_container_width=True)
            st.write(
                f"Personnel: {r['summary'].total_people} · "
                f"Violations: {r['summary'].violations} · "
                f"Status: {r['summary'].status}"
            )
            st.markdown("---")


def run_live_camera_pipeline(
    device_index: int, detect_every_n: int, base_model, ppe_model,
    person_conf: float, ppe_conf: float, requirements,
) -> None:
    """Continuously read from a local camera device and run detection at a
    throttled interval, for on-ramp testing with a physically connected
    camera (e.g. any webcam, or a photo/video camera in manufacturer webcam
    mode). Only works when this app runs locally, not on Streamlit
    Community Cloud - see the warning shown in the Live camera tab.

    This is intentionally a manual poll loop (Streamlit reruns the script
    top-to-bottom on each interaction rather than running an event loop),
    with a stop button that flips a session_state flag the loop checks.
    """
    import cv2

    st.session_state.setdefault("live_stop", False)

    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        st.error(
            f"Could not open camera at device index {device_index}. If using "
            "a camera in manufacturer webcam mode (e.g. Sony Imaging Edge "
            "Webcam, Canon EOS Webcam Utility, Fujifilm X Webcam), confirm "
            "that software is running and the camera is powered on and "
            "connected. Try a different device index if you have multiple "
            "cameras on this machine."
        )
        return

    stop_placeholder = st.empty()
    frame_placeholder_cols = st.columns(2)
    summary_placeholder = st.empty()

    if stop_placeholder.button("Stop live feed", key="stop_live"):
        st.session_state.live_stop = True

    frame_count = 0
    last_summary = None

    try:
        while not st.session_state.live_stop:
            ret, frame_bgr = cap.read()
            if not ret:
                st.warning("Lost camera feed (no frame returned). Stopping.")
                break

            run_detection_this_frame = frame_count % detect_every_n == 0

            if run_detection_this_frame:
                person_results = base_model(frame_bgr, conf=person_conf, classes=[0], verbose=False)[0]
                ppe_results = ppe_model(frame_bgr, conf=ppe_conf, verbose=False)[0]
                last_summary = summarize_results(person_results, ppe_results, requirements=requirements, frame_height=frame_bgr.shape[0])
                display_annotated = annotated_rgb(ppe_results)
            else:
                display_annotated = frame_bgr[:, :, ::-1]

            with frame_placeholder_cols[0]:
                st.image(frame_bgr[:, :, ::-1], caption="Live feed", use_container_width=True)
            with frame_placeholder_cols[1]:
                st.image(display_annotated, caption="Detections (updates every "
                          f"{detect_every_n} frames)", use_container_width=True)

            if last_summary is not None:
                v_cls = "alert" if last_summary.violations else "ok"
                summary_placeholder.markdown(
                    f'<div class="verdict {v_cls}">'
                    f"Personnel: {last_summary.total_people} · "
                    f"Violations: {last_summary.violations} · "
                    f"{last_summary.status}</div>",
                    unsafe_allow_html=True,
                )

            frame_count += 1

            # Streamlit needs to yield control periodically; without a small
            # sleep this loop can peg a CPU core and make the stop button
            # unresponsive.
            import time
            time.sleep(0.03)

    finally:
        cap.release()
        st.session_state.live_stop = False

    st.caption(
        "Live mode scores each processed frame independently, same as video "
        "mode — no cross-frame person tracking. Detection only runs every "
        f"{detect_every_n} frames to keep the preview responsive; the "
        "in-between frames show the raw feed with the last detection result "
        "still displayed as the compliance summary."
    )


def main() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">ASSIS · Phase 1</div>
            <div class="hero-title">RampGuard</div>
            <div class="hero-subtitle">Automated PPE compliance detection for airport ramp
            operations — built to run on cameras airports already have, with requirements that
            adjust to real weather conditions rather than a fixed checklist.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="feature-grid">
            <div class="feature-card">
                <div class="feature-icon">🦺</div>
                <div class="feature-title">Vest detection</div>
                <div class="feature-body">Validated on aviation ramp imagery across three
                independent test environments. The one class this system currently gates
                compliance decisions on.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">☁</div>
                <div class="feature-title">Live weather awareness</div>
                <div class="feature-body">Pulls real METAR data for any airport to adjust
                which PPE is actually required right now — not a static rulebook.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">◐</div>
                <div class="feature-title">Honest by design</div>
                <div class="feature-body">Every result is COMPLIANT, VIOLATION, or
                INDETERMINATE — the system says when it isn't sure, rather than guessing.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("What this tool can and can't do yet"):
        st.markdown(
            "- **Vest detection** is validated on real aviation ramp photos and gates "
            "compliance decisions.\n"
            "- **Helmet and glove detection** are reported for transparency but are not "
            "yet reliable enough to count as a confirmed violation — they show as "
            "INDETERMINATE, never falsely flagged.\n"
            "- **Crowded or wide-shot scenes** (many people, distant background figures) "
            "are a known limitation — role (ramp worker vs. passenger or bystander) "
            "can't currently be told apart from a bounding box alone.\n"
            "- **Camera and video certification ratings** (steel-toe boots, ANSI eye "
            "protection class) can never be verified by any camera — only visible "
            "presence is reported.\n\n"
            "Full technical detail: `docs/PPE_TAXONOMY.md`, `docs/RESEARCH_LOG.md`."
        )

    st.write("")

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
        mode = st.radio(
            "Mode",
            ["Photo analysis", "Video analysis", "Live camera (local only)"],
            horizontal=True,
            label_visibility="collapsed",
        )
        st.markdown("---")

        uploaded = None
        uploaded_video = None
        live_running = False
        device_index = 0
        detect_every_n = 10

        if mode == "Photo analysis":
            st.markdown('<span class="section-tag">Photo analysis</span>', unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "Upload a ramp / worksite photo", type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
            )
            st.caption("Analyzes a single photo. Independent from Video and Live modes below.")

        elif mode == "Video analysis":
            st.markdown('<span class="section-tag">Video analysis</span>', unsafe_allow_html=True)
            uploaded_video = st.file_uploader(
                "Upload a ramp / worksite video", type=["mp4", "mov", "avi", "m4v"],
                label_visibility="collapsed",
            )
            st.caption(
                "Video is sampled at a fixed interval (not every frame) and each "
                "sampled frame is run through the same detection pipeline as a "
                "single photo. This is frame-by-frame processing of an uploaded "
                "file, not a live camera stream — see docs/FOD_PHASE2_PLAN.md for "
                "the distinction and what live CCTV ingestion would require. "
                "Independent from Photo and Live modes."
            )

        else:  # Live camera
            st.markdown('<span class="section-tag">Live camera (local only)</span>', unsafe_allow_html=True)
            st.warning(
                "⚠ Works only when this app is running LOCALLY on a machine "
                "with a camera attached — either a built-in webcam, a "
                "standard USB webcam, or a photo/video camera running in "
                "webcam mode via its manufacturer's software (e.g. Sony "
                "Imaging Edge Webcam, Canon EOS Webcam Utility, Fujifilm X "
                "Webcam, Nikon Webcam Utility) or connected through an HDMI "
                "capture card. The deployed cloud version of this app runs "
                "on a remote server with no access to your local hardware — "
                "this mode will not work on the live public URL, only when "
                "you run `streamlit run app/streamlit_app.py` yourself on "
                "the machine the camera is connected to."
            )
            device_index = st.number_input(
                "Camera device index", min_value=0, max_value=10, value=0, step=1,
                help="0 is usually the default/built-in camera. An external "
                     "or manufacturer-webcam-mode camera is often registered "
                     "at index 1 or higher if a built-in camera is also "
                     "present on the same machine. If the feed doesn't "
                     "appear, try increasing this one at a time.",
            )
            detect_every_n = st.slider(
                "Run detection every N frames", 1, 30, 10,
                help="Higher = smoother video preview, less frequent detection "
                     "(CPU inference is the bottleneck, not the camera). Lower "
                     "= more frequent detection, choppier preview.",
            )
            live_running = st.checkbox("Start live feed")
            st.caption("Independent from Photo and Video modes.")

        if uploaded is None and uploaded_video is None and not live_running:
            st.markdown('<div class="verdict dim">— awaiting input for the selected mode —</div>', unsafe_allow_html=True)
            nothing_provided = True
        else:
            nothing_provided = False

    if nothing_provided:
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
        if live_running:
            run_live_camera_pipeline(
                device_index, detect_every_n, base_model, ppe_model,
                person_conf, ppe_conf, requirements,
            )
            return

        if uploaded_video is not None:
            run_video_pipeline(
                uploaded_video, base_model, ppe_model,
                person_conf, ppe_conf, requirements,
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
            st.markdown('<span class="section-tag dim">Original</span>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
        with col2:
            st.markdown('<span class="section-tag dim">Detections</span>', unsafe_allow_html=True)
            st.image(annotated_rgb(ppe_results), use_container_width=True)

        summary = summarize_results(person_results, ppe_results, requirements=requirements, frame_height=image_bgr.shape[0])

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
