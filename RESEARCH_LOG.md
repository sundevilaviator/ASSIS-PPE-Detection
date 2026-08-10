# ASSIS Research Log

Dated record of research and development activity. Each entry documents
concrete progress on the ASSIS platform. (Newest first.)

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
