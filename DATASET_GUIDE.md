# Dataset Guide — ASSIS PPE Detection

## Goal

1,000–3,000 labeled images, two classes:

| Class | Definition |
|---|---|
| `vest` | Person clearly wearing a high-visibility vest/jacket |
| `no-vest` | Person NOT wearing high-visibility PPE |

## Option A — Existing Roboflow Dataset (Start Here)

1. Free account at roboflow.com
2. Universe search: "PPE vest detection", "safety vest", "hi-vis detection"
3. Filter: 1,000+ images, vest-related classes, good preview quality
4. Download → YOLOv8 format → use API snippet in the Colab notebook

## Option B — Aviation-Specific Data (Phase 2)

Sources: your own authorized photos (written permission; NEVER SIDA/secure
areas), FAA/NTSB public archives, airport press kits, licensed video frames.

**Legal guardrails:**
- NEVER use CCTV footage without a formal data-sharing agreement
- NEVER photograph SIDA or sterile areas (TSA violation)
- Blur faces in anything you publish
- When in doubt: synthetic or public data only

## Annotation Workflow

1. Upload to Roboflow project
2. Box every person; label `vest` or `no-vest`
3. Include full torso in each box, consistently
4. Use augmentation (flip, brightness ±25%, slight rotation) for 2–3x

## Quality Checklist

- [ ] 500+ images per class
- [ ] Mixed distances, lighting, angles, occlusions
- [ ] 70/20/10 train/val/test split
- [ ] No duplicates across splits
