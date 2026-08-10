# White Paper Outline — ASSIS Phase 1

**Working title:** "Transfer Learning for Aviation Ramp Safety: A YOLOv8-Based
Approach to Automated PPE Compliance Monitoring at U.S. Airports"

**Target venues:** 1) TRB Annual Meeting (deadline ~Aug 1 annually),
2) AAAE technical publications, 3) arXiv preprint (immediate)

**Length:** 8–12 pages

## Sections

1. **Abstract** (write LAST) — problem, approach, key metric, implication
2. **Introduction** (write FIRST) — ramp injury/cost statistics; FAA Part 139
   SMS, OSHA 1910.132, ICAO Annex 19 context; the monitoring gap; research
   question: can transfer learning from general PPE data reach deployable
   accuracy for aviation?
3. **Background & Related Work** — commercial systems (Xovis, Bosch IVA, ADB
   Safegate) scope limits; construction-domain PPE CV literature; the
   aviation-specific integration gap; ACRP SMS context for small airports
4. **Methodology** — YOLOv8n choice, dataset, augmentation, training setup
   (Colab T4, epochs, hyperparameters), metrics, compliance scoring logic
5. **Results** — metrics table, training curves, qualitative examples,
   inference speed, honest failure analysis
6. **Discussion** — deployment feasibility, SMS workflow integration,
   privacy/SSI-avoiding design, limitations
7. **Future Work** — FOD module, pose estimation, real-time pipeline, pilot
   study design with small/non-hub airports (ACRP linkage)
8. **Conclusion** — one paragraph
9. **References** — FAA AC 150/5200-37, 14 CFR 139, OSHA 1910.132,
   Ultralytics docs, ACRP reports, 5–10 academic PPE-detection citations

## Writing Schedule

| Week | Section |
|---|---|
| 1–2 | Intro + Background |
| 3–6 | Methodology (as you train) |
| 7–8 | Results |
| 9–10 | Discussion + Conclusion + Abstract |
| 11–12 | Review, format, submit |
