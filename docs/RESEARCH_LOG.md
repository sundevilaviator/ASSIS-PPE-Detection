# ASSIS Research Log

Dated record of research and development activity. Each entry documents concrete progress on the ASSIS platform. (Newest first.)

## 2026-08-13 — Critical inference bug: BGR/RGB channel order

**The defect.** The Streamlit demo passed an RGB numpy array to the YOLOv8 model. Ultralytics treats numpy arrays as BGR by convention, so the model was seeing colour-inverted pixels — high-visibility yellow rendered as blue. Detection failed silently, with no error at any stage.

**Evidence** (night ramp-marshalling photograph, `models/best.pt`, conf=0.4):

| Input method | Detections |
|---|---|
| File path | `vest 0.92` |
| PIL image | `vest 0.92` |
| numpy RGB (the defective app path) | none |
| numpy BGR (corrected) | `vest 0.92` |

This explains a discrepancy that had looked inconsistent up to this point: Colab evaluation passed a file path and worked fine, while the deployed app passed an RGB array and failed on the identical photo.

**Fixes applied:**
- `app/streamlit_app.py` now converts RGB to BGR before inference, and reverses `plot()`'s output back to RGB for display. This also explains the inverted colours visible in earlier demo screenshots.
- Added `tests/test_channel_order.py` — three regression tests pinning the input convention, with a committed fixture image. All pass.
- Repaired `src/detect.py`, which was calling `summarize_results()` with one argument instead of two and reading a `.compliant` attribute that doesn't exist. It crashed on every invocation. It now uses the same dual-model path as the app.
- `@st.cache_resource` is now keyed on a SHA-256 fingerprint of `models/best.pt`, so swapping the weights invalidates the cache instead of silently serving a stale model. Added a build-diagnostics panel (ultralytics/torch versions, weights hash) for comparing local runs against the deployed one.
- Pinned ultralytics to `8.4.118` in `requirements.txt` for reproducibility.

**Re-evaluation after the fix.** Vest detects reliably on aviation ramp imagery — 0.883 and 0.920 on two held-out photographs, correctly matched to the person in frame. Helmet and gloves still don't detect on the same imagery even at conf=0.05, which confirms this is a real limitation and not an artifact of the channel bug. Helmet was trained on rigid construction hard hats, which aren't standard ramp PPE — ramp crews wear caps and hearing protection instead. Gloves was already the weakest class in-domain (0.804 mAP50) and degrades further at the scale typical of wide-angle apron photography.

**Correction to the prior entry.** The 2026-08-12 entry described "cross-domain validation" without noting that it covered vest only, and that it ran through a file-path code path not equivalent to the deployed app. Noting both here for the record.

**Deployment verification.** The fix is live and confirmed against the held-out night photo: vest 0.92, one person detected, zero violations, 100% compliance, correct colour rendering. The deployed result now matches local inference, closing the gap that originally surfaced this bug.

**Next:** re-scope helmet toward hearing-protection detection, which better matches actual ramp PPE; collect aviation-specific glove imagery at realistic camera distances for Phase 2 fine-tuning; build an annotated aviation validation set (target 100+ images) so precision/recall can be reported instead of single-image confidences; begin FOD module scoping.

## 2026-08-13 (continued) — False positive, UI redesign, product naming

**False positive.** A test on a non-ramp scene — two people at a terminal curb, no PPE context — returned a vest detection at 0.44 confidence on a plain shirt. Genuine detections elsewhere in this project have consistently landed at 0.88–0.92, so this wasn't edge-case noise; it pointed to a real precision gap in the 0.40–0.55 band. Fixed by raising the default PPE threshold from 0.40 to 0.60, and the UI now flags anything below 0.70 as "unverified" in the raw-detections table regardless of where the slider is set, so a lowered threshold can't quietly present a weak detection as confirmed.

**UI redesign.** Went through three iterations based on direct feedback. First was a dark "Glass Cockpit" aviation-instrument theme. Feedback that the phosphor-green-on-black look read as a hacker terminal rather than professional software led to a second pass with a muted slate/sage palette. The final version is a BLADE-inspired light-first design with a working light/dark toggle, built against an explicit brief covering hex palette, responsive split-panel layout, and outdoor-glare accessibility. Rationale for each pass is in `docs/UI_UX_BLUEPRINT.md`. The METAR panel moved from the sidebar to a sticky primary panel on desktop (stacking below the imagery panel on mobile) after testing showed it was easy to miss where it started.

**Product naming.** Named the PPE module "ASSIS — Sentry," then checked it against USPTO's trademark database and found a conflict with an active, enforced mark (Functional Software Inc.'s Sentry) plus a second live mark, "Watchtower," with closer category overlap. Landed on "RampGuard" after a direct TESS search: a 1982 application for that name (Foxtronics, for an aircraft-monitoring device) was abandoned for failure to respond and never registered, with nothing since. Adopted "ASSIS — RampGuard" as the module name, keeping "ASSIS" as the umbrella project name throughout, consistent with everything already filed.

**Next:** gloves retraining (scaffolded, not yet run); continue TRB/ACRP outreach for an institutional letter.

## 2026-08-13 (continued) — Phase 2 (FOD) scoping, vendor classification, prototype scaffold

**Initial scoping.** Drafted `docs/FOD_PHASE2_PLAN.md`. FOD detection is a meaningfully harder problem than PPE — smaller objects, lower contrast, no anchor object to correlate against, and less public training data. Recommended starting narrow: large, obvious debris only.

**Positioning correction.** A follow-up review turned up a real classification error in the competitive landscape section. Stratech's iFerret and ArgosAI's A-FOD had been grouped with the radar vendors (Xsight, QinetiQ, Rheinmetall). Both are actually dedicated electro-optical, camera-only systems, already deployed commercially at major hubs — Changi, Dubai, Heathrow, Miami, among others. That invalidated the original differentiator ("camera-based, unlike the radar vendors"), since camera-only FOD detection turns out to be an established category, not something novel. Repositioned around existing-camera, apron-focused FOD triage integrated with the PPE ingestion pipeline — the differentiator is deployment context and infrastructure reuse, not sensing modality. Propagated the same correction into the white paper (Section 2.4 and the conclusion), which had made the identical now-incorrect claim independently.

**Dataset resolution.** The planning doc had assumed no large public FOD dataset existed. That was wrong. FOD-A (Munyer et al., 2021, IEEE ICMLA; arXiv:2110.03072) is real, peer-reviewed, and directly downloadable — 31 object categories, 30,000+ annotated instances, with light-level and weather sub-labels. Confirmed via GitHub (`FOD-UNOmaha/FOD-data`) and a Kaggle mirror.

**Prototype scaffold.** Built `notebooks/ASSIS_FOD_Phase2_Prototype.py`, applying the Phase 1 small-object lessons from the start rather than rediscovering them — `yolov8s` instead of nano, `imgsz=960` instead of 640, both taken directly from the PPE gloves/helmet resolution finding. Nothing has been run yet; no FOD model has been trained or evaluated. This is a scoping and tooling step, not a results step.

**Further grounding.** Identified and cited nine peer-reviewed FOD-detection papers from 2021–2026, establishing that YOLO-family models are viable for this task under controlled research conditions. Noted explicitly that this is feasibility evidence, not a guarantee of performance in ASSIS's actual deployment context — existing CCTV, variable geometry, variable lighting.

**Next:** Phase 2a camera survey (per `docs/FOD_PHASE2_PLAN.md` §5.1) — a low-cost prerequisite that determines whether existing apron cameras can even resolve FOD-relevant object sizes before any model work is worth doing. Requires written authorization from a host airport's operations/security management before any camera access or on-site work; not yet sought.

## 2026-08-13 (continued) — METAR conditional policy layer

**METAR-conditional PPE policy.** Built `src/conditions.py`, which pulls live METAR observations (NOAA Aviation Weather Center, no API key needed) for a given airport and derives which PPE classes are currently required from wind speed, temperature, and reported weather — rather than applying one fixed rule set. Falls back to the stricter requirement set if the observation is missing or older than 90 minutes. Verified live against a real airport code (KCHS): correctly returns wind, temperature, and raw METAR text, and correctly falls back to defaults when no station is set.

**Fixes shipped this session so far:**
- BGR/RGB channel-order defect (silent vest-detection failure in deployment) — fixed and regression-tested
- False-positive vest detection at 0.44 confidence — default threshold raised to 0.60, weak detections flagged unverified in the UI
- Stale-cache risk on model reload — cache now keyed to the weights file's SHA-256
- FOD vendor-classification error (Stratech/ArgosAI wrongly grouped as radar) — corrected in `docs/FOD_PHASE2_PLAN.md` and `docs/WHITE_PAPER_DRAFT.md`

**Next phase.** Phase 2 scoping is done, with a working prototype scaffold (`notebooks/ASSIS_FOD_Phase2_Prototype.py`) built around the FOD-A dataset. Dataset acquisition, annotation conversion, and initial training can proceed without airport-specific access. The camera survey (Phase 2a) requires written authorization from a host airport before any camera access or on-site work — not yet sought or obtained — and is not currently blocking the dataset/model track. Gloves retraining for Phase 1 is still scaffolded but not run.

## 2026-08-13 (continued) — Video mode, size-filter calibration, inference-resolution finding (final PPE Phase 1 entry)

**Video and live-camera modes.** Extended the app past single-photo upload: video files are now sampled frame-by-frame (one frame every two seconds), and a local live-camera mode supports testing with a physically connected camera — this only works when the app is run locally, since a remote cloud instance has no access to local hardware. After feedback that sharing tabs made photo, video, and live modes feel intermingled, restructured them into fully separate top-level modes.

**Crowded-scene false-violation finding, and a mistake in the first fix.** A real wide-shot apron video (81.5s, 1920x1080, several people) initially produced 89 violations across 30 sampled frames. The cause: the generic COCO person detector has no concept of a ramp worker versus a background bystander or passenger, so it was flagging everyone in frame. Added a minimum-person-size filter (`MIN_PERSON_HEIGHT_FRACTION` in `src/utils.py`) to exclude small, distant detections. The first value chosen — 0.15, meaning a person had to occupy 15% of frame height — was too aggressive. Retested against the same video and it excluded every person in frame, including genuinely vested workers, producing zero personnel detected. That's a worse failure than the one it was meant to fix. Recalibrated to 0.03 after measuring the actual box heights in the footage (roughly 9-11% for workers at realistic ramp distance), retested, and confirmed correct this time. Also added diagnostic tooling directly to the UI — raw versus filtered versus deduplicated counts, plus a checkbox to disable the filter entirely — so a miscalibration like this is self-diagnosable next time rather than needing another multi-round back-and-forth.

**Root cause: inference resolution was silently defaulting to 640px on 1080p video.** Even after the size-filter fix, the same video returned zero vest detections despite a clearly visible, human-identifiable hi-vis vest in frame — confirmed by cropping and inspecting the frame directly. Testing across `imgsz` values on that exact frame:

| imgsz | Vest detection |
|---|---|
| 640 (the previous silent default - never explicitly set anywhere in the app) | none |
| 960 | none |
| 1280 | 0.849 confidence |
| 1920 (native) | 0.862 confidence |

This is the same small-object/resolution issue first found with helmet detection back in the 2026-08-12 entry, now confirmed to affect vest too under realistic wide-shot or elevated-CCTV geometry, not just close-up photos. `INFERENCE_IMGSZ = 1280` is now set explicitly across all three pipelines - photo, video, live camera. Cost: roughly 3.6x slower per frame on CPU (165ms to 592ms), which is acceptable for single photos and sampled video.

**End-to-end check on the real video.** With both fixes in place, the first 10 sampled frames produced 7 vest-compliant detections out of 57 total person-detections and 50 violations - versus zero compliant detections before either fix. Whether that violation count is itself accurate hasn't been independently checked yet. It could reflect genuine non-compliance, bystanders being miscounted as personnel (the crowded-scene issue noted above, still unresolved), or some mix of both. Telling those apart needs someone to look at the flagged frames; it isn't something a code change can settle on its own.

**Where this leaves Phase 1 PPE work for this session.** Gloves detection is still unresolved - retraining is scaffolded but hasn't run. Helmet is still a taxonomy mismatch pending re-scoping to hearing protection. A labeled aviation validation set is still needed before any accuracy claim beyond single-image or single-video spot checks. Moving focus to Phase 2 from here.

## 2026-08-13 (continued) — AttributeError traced to a module-name collision, not a stale deployment

A live deployment kept throwing `AttributeError` on `ComplianceSummary.total_person_detections_unfiltered`, even after confirming the file containing that field was pushed and the app rebooted - same error, unchanged, across three separate fix-and-verify attempts. The actual cause: `src/utils.py` was being imported via `sys.path.insert(0, ...)` followed by `from utils import ...`, which is fragile because Python checks `sys.modules` before it ever looks at `sys.path`. `utils` is a generic name. If anything in the dependency chain - streamlit, ultralytics, torch, or something underneath any of them - had already registered a module under that same name, the `sys.path` insert would be silently ignored and the wrong module handed back, with no error and no indication anything was wrong. Just missing attributes at runtime.

**Verification.** Rather than take the diagnosis on faith, it was tested directly: planted a fake `utils` module in `sys.modules` before import, recreating the suspected collision, and confirmed the old import method really would have been vulnerable to it. The fix - loading `src/utils.py` and `src/conditions.py` by explicit file path through `importlib.util`, registered under collision-proof names (`assis_utils`, `assis_conditions`) - was then confirmed to load the correct module even with the fake one still in place, and confirmed to behave normally with no regression in a clean environment.

**How confident is this diagnosis, really.** The fix is verified sound against the specific collision it targets. Whether module-name collision was definitely what caused the original deployed error - as opposed to, say, a deployment sync issue that happened to resolve around the same time - wasn't independently confirmed against the actual live environment. The fix stands regardless, since it closes off an entire class of fragile-import failure by construction and costs nothing to keep.

## 2026-08-13 (continued) — Crowded-scene vest confidence ceiling: a real detection limit, not a tuning problem

Two more findings came out of follow-up testing, one cosmetic and one substantive, and together they close out further threshold-tuning on PPE for this session.

**Cosmetic: overlapping duplicate vest boxes on one person.** A short test video of a lone worker showed three overlapping "vest" boxes (0.90, 0.92, 0.96) drawn on the same person. Checked whether this affected the actual compliance decision - it didn't. The frame correctly reported one person, zero violations, compliant, because the correlation logic only needs one qualifying overlap, not exactly one box. Applied a stricter NMS IoU (`PPE_NMS_IOU = 0.4`, down from ultralytics' default 0.7) purely for visual cleanliness, and confirmed no regression against the known-good single-detection case. Worth being clear that this was a display fix, not a logic fix - the underlying answer was already right.

**Substantive: dense, wide-shot scenes push real vest confidence below the working threshold, and there's no threshold that fixes this without reopening the earlier false-positive problem.** Dug into the specific frame that produced a 5-violation, zero-compliant result (t=15.7s in the crowded gate video) and found a genuine vest signal - but only at 0.451 confidence, squarely inside the same 0.40-0.55 band already flagged as unreliable from the earlier false-positive finding. Lowering the threshold enough to catch this would reopen that exact risk. There's no single threshold value that recovers dense-scene detections without also readmitting misclassifications elsewhere. This looks like a genuine detection-quality ceiling for this kind of scene - small, partially occluded objects at wide-shot or elevated-camera distance - rather than something misconfigured. Tested it from four separate angles this session (resolution, minimum-size filtering, NMS, confidence threshold), and all four point the same way.

**Decision: stop tuning against this for now.** Further threshold adjustment has hit diminishing returns and risks trading one documented problem for another. An actual fix would need a different approach - closer or higher-resolution camera coverage, tiled sub-region inference at higher effective resolution, or retraining specifically on dense small-object footage. Each of those is a real project, not a settings change. Recording this as a characterized, honest Phase 1 boundary and moving on to Phase 2.

---

## 2026-08-12 — 3-class retrain, dual-model violation detection, cross-domain validation

- Simplified the dataset from 5 classes to 3: dropped `boots` and `human`
- Retrained YOLOv8n on 3 classes (gloves, helmet, vest), 50 epochs, Colab Pro T4
- Results: mAP50 0.922, mAP50-95 0.725, precision 0.922, recall 0.864
- Per-class mAP50: vest 0.985, helmet 0.978, gloves 0.804
- Built dual-model violation detection (`src/utils.py`): correlates person detections with PPE detections via bounding-box overlap for true per-person compliance
- Cross-domain check: tested on two real aviation ramp photos never seen in training. Both correctly showed vest presence (0.92 and 0.80 confidence) despite training on non-aviation imagery only
  - **[Amended 2026-08-13]** This result covers vest only, and came from passing a file path to the model. It doesn't say anything about helmet or gloves, and wasn't equivalent to the deployed app's actual code path, which had the channel-order defect described in the 2026-08-13 entry.
- Updated license to AGPL-3.0; added NOTICE.md, CITATION.cff, CLA
- Next: Phase 2 (FOD detection) dataset scoping

---

## 2026-08-10 — First model training run: mAP50 = 0.933

- Dataset: PPE-Detection (Roboflow Universe), 5 classes (boots, gloves, helmet, human, vest), 1,566 training images, 420 validation images
- Training: YOLOv8n, 50 epochs, Google Colab T4
- Results: mAP50 0.933 (past the 0.85 MVP target), mAP50-95 0.728, precision 0.924, recall 0.893
- Per-class mAP50: vest 0.976, helmet 0.984, human 0.957, boots 0.933, gloves 0.769
- Ran inference on sample imagery and confirmed the detection pipeline works
- Downloaded the trained weights (`best.pt`) into the repo
- Gloves is noticeably weaker (0.769) than the other classes - likely smaller object size and less representation in the dataset. Flagged as a Phase 1.5 target.
- Next: test on aviation/ramp imagery for cross-domain transfer; update compliance-scoring logic for the full 5-class output; put these numbers into the white paper's Results section

---

## 2026-08-09 — Repository published on GitHub

- Created the public repo: github.com/sundevilaviator/ASSIS-PPE-Detection
- Uploaded the full Phase 1 codebase and reorganized it into `app/`, `src/`, `docs/`, `notebooks/`
- 14 commits covering iterative setup and corrections
- Next: dataset selection (Roboflow) and first training run

---

## 2026-08-08 — Repository established; Phase 1 MVP scaffolded

- Created the public repo with the complete Phase 1 codebase
- Components: YOLOv8 training pipeline (Colab notebook plus CLI script), compliance-scoring module, Streamlit demo app
- Defined v1 detection scope: high-visibility vest compliance (vest / no-vest) for ramp operations
- Documented the dataset acquisition strategy - Roboflow Universe PPE datasets, transfer learning from construction-domain imagery
- Documented regulatory alignment: FAA 14 CFR Part 139 (SMS), TSA 49 CFR Part 1542, ICAO Annex 19, OSHA 29 CFR 1910.132
- Next: dataset selection and first training run (target mAP50 >= 0.85)

<!-- Template for future entries:

## YYYY-MM-DD — Title

- What was done
- Results/metrics if any
- Next step
-->
