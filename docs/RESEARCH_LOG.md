# ASSIS Research Log

Dated record of research and development activity. Each entry documents
concrete progress on the ASSIS platform. (Newest first.)

## 2026-08-13 — Critical inference bug identified and fixed (BGR/RGB channel order)

**Defect.** The Streamlit demo application passed an **RGB** numpy array to
the YOLOv8 model. Ultralytics interprets numpy array inputs as **BGR**, so
the model received colour-inverted pixels — high-visibility yellow rendered
as blue. Detection degraded severely with no error raised.

**Evidence** (night ramp-marshalling photograph, `models/best.pt`, conf=0.4):

| Input method | Detections |
|---|---|
| File path | `vest 0.92` |
| PIL image | `vest 0.92` |
| numpy RGB (defective app path) | *none* |
| numpy BGR (corrected) | `vest 0.92` |

This reconciles a previously unexplained discrepancy: Colab evaluation passed
a file path and succeeded, while the deployed application passed an RGB array
and failed on the identical image.

**Fixes applied.**
- `app/streamlit_app.py`: convert RGB → BGR before inference; `plot()` output
  reversed back to RGB for display (this also resolves the inverted colours
  visible in earlier demo screenshots).
- `tests/test_channel_order.py`: three regression tests pinning the input
  convention, with a committed fixture image. All passing.
- `src/detect.py`: repaired — it called `summarize_results()` with one
  argument (two required) and read a non-existent `.compliant` attribute; the
  CLI crashed on every invocation. Now uses the same dual-model path as the app.
- `app/streamlit_app.py`: `@st.cache_resource` now keyed on a SHA-256
  fingerprint of `models/best.pt`, so replacing the weights invalidates the
  cache instead of silently serving a stale model. Build diagnostics panel
  added (ultralytics/torch versions, weights hash) for local-vs-deployed
  comparison.
- `requirements.txt`: ultralytics pinned to `8.4.118` for reproducibility.

**Re-evaluation after fix.** `vest` detects reliably on aviation ramp imagery
(0.883 and 0.920 on two held-out photographs, correctly correlated to
personnel). `helmet` and `gloves` remain undetected on the same imagery even
at conf=0.05 — confirmed as a genuine limitation *after* eliminating the
channel-order confound, not an artefact of it. Assessed causes: the `helmet`
class was trained on rigid construction hard hats, which are not standard ramp
PPE (ramp personnel wear caps and hearing protection); `gloves` was already
the weakest class in-domain (0.804 mAP50) and degrades further at the object
scale typical of wide-angle apron photography.

**Correction to prior entry.** The 2026-08-12 entry described "cross-domain
validation" without recording that it covered the `vest` class only, and that
it was performed via a file-path code path not equivalent to the deployed
application. Both qualifications are noted here for accuracy.

**Deployment verification.** Fix deployed to the public Streamlit application
and confirmed on the held-out night ramp-marshalling photograph: `vest 0.92`,
1 person detected, 0 violations, 100% compliance, with correct colour
rendering in the annotated output. Deployed result now matches local
inference exactly, closing the local-versus-deployed discrepancy that
originally surfaced this defect.

**Next.** (1) Re-scope `helmet` toward hearing-protection detection, which
reflects actual ramp PPE requirements; (2) collect aviation-specific glove
imagery at representative camera distances for Phase 2 fine-tuning;
(3) assemble an annotated aviation validation set (target 100+ images) to
support reportable precision/recall figures rather than single-image
confidences; (4) begin FOD module scoping.

## 2026-08-13 (continued) — False-positive finding, UI/UX redesign, and product naming

**False-positive finding.** A live test on a non-ramp, landside scene (two people at a terminal curb, no PPE context) produced a vest detection at 0.44 confidence -- a plain shirt misread as a vest. Cross-referenced against confirmed genuine detections across this project (consistently 0.88-0.92), this revealed a real precision gap in the 0.40-0.55 confidence band, not edge-case noise. Fix: default PPE confidence threshold raised from 0.40 to 0.60; the UI now flags any detection below 0.70 as "unverified" in the raw-detections table regardless of the active threshold, so a lowered slider can't silently present a weak detection as confirmed.

**UI/UX redesign.** The Streamlit interface went through three design iterations in response to direct feedback: (1) an initial dark "Glass Cockpit" aviation-instrumentation theme, (2) a correction after feedback that the phosphor-green-on-black look read as a hacker terminal rather than professional software, moved to a muted slate/sage palette, (3) a final BLADE-inspired light-first redesign with a working light/dark toggle, per an explicit design brief (hex palette, responsive split-panel layout, outdoor-glare accessibility requirements). Design rationale for each iteration is recorded in `docs/UI_UX_BLUEPRINT.md`. The METAR weather panel was relocated from the sidebar to a primary, sticky-positioned panel on desktop (stacking below the imagery panel on mobile) after user testing showed it was easy to miss in its original location.

**Product naming.** The PPE module was named "ASSIS -- Sentry," then checked against USPTO's public trademark search and found to conflict with an active, enforced registered mark (Functional Software, Inc.'s Sentry, a widely-used software product) and a second live mark ("Watchtower," closer in category overlap). A third candidate, "RampGuard," was verified via direct USPTO TESS search: a 1982 trademark application (Foxtronics, for an aircraft-monitoring device) was abandoned for failure to respond and never registered, with no subsequent filings found. "ASSIS -- RampGuard" was adopted as the module name; "ASSIS" remains the umbrella project name throughout, consistent with all previously filed documents.

**Next:** gloves retraining (scaffolded, not yet executed); continue TRB/ACRP outreach for institutional letter.

## 2026-08-13 (continued) — Phase 2 (FOD) scoping, vendor classification, and prototype scaffold

**Initial scoping.** Drafted `docs/FOD_PHASE2_PLAN.md`, identifying FOD detection as a substantially harder problem than PPE (small object size, low contrast, no anchor object, scarce public training data) and recommending a narrow first scope (large/obvious debris only).

**Positioning correction.** A follow-up review found a material classification error in the initial competitive landscape: Stratech's iFerret and ArgosAI's A-FOD were incorrectly grouped with radar-based vendors (Xsight, QinetiQ, Rheinmetall). Both are dedicated electro-optical (camera-only) systems, commercially deployed at major hubs (Changi, Dubai, Heathrow, Miami, and others). This invalidated the original differentiator claim ("camera-based, unlike radar vendors"), since camera-only FOD detection is an established, deployed category, not a novel approach. The positioning was corrected to: existing-airport-camera, apron/ramp-focused FOD triage, integrated with the existing PPE ingestion pipeline -- differentiated by deployment context and infrastructure reuse, not by sensing modality. This correction was propagated to the white paper (Section 2.4 and Conclusion), which had independently made the same now-corrected claim.

**Dataset resolution.** The initial planning document's assumption that "no large public FOD dataset exists" was found to be incorrect. FOD-A (Munyer et al., 2021, IEEE ICMLA; arXiv:2110.03072) is a real, peer-reviewed, directly downloadable dataset -- 31 object categories, 30,000+ annotated instances, with light-level and weather sub-labels. Source confirmed via GitHub (`FOD-UNOmaha/FOD-data`) and a Kaggle mirror.

**Prototype scaffold.** Built `notebooks/ASSIS_FOD_Phase2_Prototype.py`, applying Phase 1's small-object lessons from the outset rather than rediscovering them: `yolov8s` (not nano) and `imgsz=960` (not 640) as starting parameters, based directly on the PPE gloves/helmet resolution finding. Not yet executed -- no FOD model has been trained or evaluated. This is a scoping and tooling milestone, not a results milestone.

**Additional technical grounding.** Nine peer-reviewed FOD-detection papers (2021-2026) identified and cited, establishing YOLO-family feasibility for FOD detection in controlled research conditions -- explicitly noted as feasibility evidence, not a guarantee of performance in ASSIS's actual deployment context (existing CCTV, varied geometry and lighting).

**Next:** Phase 2a camera survey at CHS (per `docs/FOD_PHASE2_PLAN.md` Section 5.1) -- a prerequisite, low-cost step that determines whether existing apron cameras can support FOD detection at all before further model work is justified.

## 2026-08-13 (continued) — METAR conditional policy layer

**METAR-conditional PPE policy.** Implemented `src/conditions.py`, which queries live METAR observations (NOAA Aviation Weather Center, no API key required) for a given airport and derives which PPE classes are currently required based on wind speed, temperature, and reported weather phenomena, rather than applying a fixed requirement set. Fails safe to the stricter requirement set if the observation is unavailable or older than 90 minutes. Verified live in the deployed app against a real airport code (KCHS), correctly returning wind, temperature, and raw METAR text, and correctly falling back to defaults when no station is set.

**Summary of fixes this session (2026-08-13):**
- BGR/RGB channel-order defect (silent vest-detection failure in deployment) — fixed, regression-tested
- False-positive vest detection at 0.44 confidence on a non-ramp scene — default threshold raised to 0.60, weak detections flagged as unverified in the UI
- Stale-cache risk on model reload — cache now keyed to weights file SHA-256
- FOD Phase 2 vendor-classification error (Stratech/ArgosAI incorrectly grouped as radar vendors) — corrected across `docs/FOD_PHASE2_PLAN.md` and `docs/WHITE_PAPER_DRAFT.md`

**Next phase.** Phase 2 (FOD detection) scoping is complete with a working prototype scaffold (`notebooks/ASSIS_FOD_Phase2_Prototype.py`) using the FOD-A dataset. Immediate next step is Phase 2a: a camera survey at CHS to determine whether existing apron cameras can resolve FOD-relevant object sizes before any model training is justified. Gloves retraining for Phase 1 remains scaffolded but not executed.

## 2026-08-13 (continued) — Video mode, size-filter calibration, and inference-resolution finding (final PPE Phase 1 entry)

**Video and live-camera modes added.** Extended the app beyond single-photo upload: video file upload (sampled frame-by-frame, one frame every 2 seconds), and a local live-camera mode for physically-connected-camera testing (works only when run locally, not on the deployed cloud instance, since a remote server cannot access local hardware). Photo, video, and live modes were restructured from shared tabs into fully separate top-level modes after user feedback that shared UI made the three inputs feel intermingled rather than independent.

**Crowded-scene false-violation finding, and a self-correction on the fix.** Testing against a real wide-shot apron video (81.5s, 1920x1080, multiple people) initially produced 89 violations from 30 sampled frames — traced to the generic COCO person detector having no concept of "ramp worker" vs. "background bystander/passenger." A minimum-person-size filter was added to exclude small/distant detections (`MIN_PERSON_HEIGHT_FRACTION` in `src/utils.py`). The first calibration (0.15, i.e. people must occupy 15% of frame height) was too aggressive: retested against the same real video, it excluded every person in frame, including genuine vested ramp agents, producing 0 personnel detected — a worse failure than the problem it was meant to fix. Recalibrated to 0.03 after measuring actual box-height fractions in the real footage (~9-11% for legitimate ramp-distance workers); retested and confirmed correct. Diagnostic tooling was added directly to the UI (raw vs. filtered vs. deduplicated counts, plus an explicit checkbox to disable the filter) specifically so this class of miscalibration is self-diagnosable next time, rather than requiring another multi-round back-and-forth to isolate.

**Root-cause finding: inference resolution silently defaulted to 640px on 1080p video.** After the size-filter fix, the same test video still returned zero vest detections despite a clearly visible, human-identifiable hi-vis vest in frame (confirmed by manually cropping and inspecting the actual frame). Direct testing across `imgsz` values on the identical frame:

| imgsz | Vest detection |
|---|---|
| 640 (previous silent default - never explicitly set anywhere in the app) | none |
| 960 | none |
| 1280 | 0.849 confidence |
| 1920 (native) | 0.862 confidence |

This is the same small-object/resolution finding first identified for helmet detection in the 2026-08-12 entry, now confirmed to affect vest detection as well under realistic wide-shot/elevated-CCTV camera geometry - not just close-up photography. `INFERENCE_IMGSZ = 1280` is now set explicitly across all three pipelines (photo, video, live camera) in `app/streamlit_app.py`. Measured cost: ~3.6x slower per frame on CPU (165ms to 592ms), judged acceptable for single-photo and sampled-video use.

**End-to-end verification on the real video.** With both fixes applied, the first 10 sampled frames of the test video produced 7 vest-compliant detections (versus 0 before either fix) out of 57 total person-detections, with 50 violations. The violation count itself is not yet independently verified as accurate - it may reflect genuine non-compliance, bystanders/passengers incorrectly counted as personnel (the crowded-scene limitation noted above, not yet resolved), or some combination; distinguishing these requires visual review of the flagged frames, which is a manual task, not a code fix.

**Status: this closes active Phase 1 (PPE) development for this session.** Known open items, unchanged from prior entries: gloves detection remains unresolved (scaffolded retraining not yet executed), helmet remains a taxonomy mismatch pending re-scoping to hearing protection, and a labeled aviation validation set is still required before any accuracy claim beyond single-image/single-video spot checks. Development focus moves to Phase 2 (FOD).

---

## 2026-08-12 — 3-class retrain, dual-model violation detection, cross-domain validation

- Simplified dataset from 5 classes to 3: removed `boots` and `human`
- Retrained YOLOv8n (3 classes: gloves, helmet, vest), 50 epochs, Colab Pro T4 GPU
- Results: mAP50 0.922, mAP50-95 0.725, Precision 0.922, Recall 0.864
- Per-class mAP50: vest 0.985, helmet 0.978, gloves 0.804
- Implemented dual-model violation detection (src/utils.py): correlates
  person detections with PPE detections via bounding-box overlap for true
  per-person compliance scoring
- Cross-domain validation: tested on two real aviation ramp photographs
  (never seen in training). Both correctly detected vest presence
  (confidence 0.92 and 0.80) despite training on non-aviation imagery only
  - **[Amended 2026-08-13]** This result covers the `vest` class only, and was
    obtained by passing a file path to the model. It does not characterise
    `helmet` or `gloves`, and was not equivalent to the deployed application's
    code path, which carried the channel-order defect described in the
    2026-08-13 entry.
- License updated to AGPL-3.0; added NOTICE.md, CITATION.cff, CLA
- Next: Phase 2 (FOD detection) dataset scoping

---

## 2026-08-10 — First model training completed: mAP50 = 0.933

- Dataset: PPE-Detection (Roboflow Universe), 5 classes (boots, gloves,
  helmet, human, vest), 1,566 training images, 420 validation images
- Training: YOLOv8n, 50 epochs, Google Colab T4 GPU
- Results:
  - mAP50: 0.933 (exceeds MVP target of ≥ 0.85)
  - mAP50-95: 0.728
  - Precision: 0.924
  - Recall: 0.893
- Per-class mAP50: vest 0.976, helmet 0.984, human 0.957, boots 0.933,
  gloves 0.769
- Tested inference on sample imagery; confirmed working detection pipeline
- Trained weights (best.pt) downloaded and added to repo
- Limitation observed: gloves class shows lower accuracy (0.769) than
  other classes — likely due to smaller object size and dataset
  representation; noted as a target for improvement in Phase 1.5
- Next: test on aviation/ramp-context imagery to evaluate cross-domain
  transfer; update compliance-scoring logic for full 5-class output;
  populate white paper Results section with these metrics

---

## 2026-08-09 — Repository published and organized on GitHub

- Created public GitHub repository: github.com/sundevilaviator/ASSIS-PPE-Detection
- Uploaded full Phase 1 codebase and reorganized into proper folder structure
  (app/, src/, docs/, notebooks/)
- 14 commits reflecting iterative setup and corrections
- Next: dataset selection (Roboflow) and first model training run

---

## 2026-08-08 — Repository established; Phase 1 MVP scaffolded

- Created public repository with complete Phase 1 codebase
- Components: YOLOv8 training pipeline (Colab notebook + CLI script),
  compliance-scoring module, Streamlit demonstration application
- Defined v1 detection scope: high-visibility vest compliance
  (`vest` / `no-vest`) for ramp operations
- Documented dataset acquisition strategy (Roboflow Universe PPE datasets;
  transfer learning from construction-domain imagery)
- Documented regulatory alignment: FAA 14 CFR Part 139 (SMS),
  TSA 49 CFR Part 1542, ICAO Annex 19, OSHA 29 CFR 1910.132
- Next: dataset selection and first training run (target mAP50 ≥ 0.85)

<!-- Template for future entries:

## YYYY-MM-DD — Title

- What was done
- Results/metrics if any
- Next step
-->
