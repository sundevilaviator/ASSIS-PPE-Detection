# ASSIS Research Log

Dated record of research and development activity. Each entry documents
concrete progress on the ASSIS platform. (Newest first.)

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
