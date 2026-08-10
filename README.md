# ASSIS — PPE Detection Module (MVP)

**AI-Integrated Airport Safety and Security Intelligence System**
*Phase 1: Personal Protective Equipment (PPE) Compliance Detection for Airport Ramp Operations*

---

## Overview

ASSIS is a research initiative to develop an integrated, AI-driven safety and
security intelligence platform for U.S. airports. This repository contains
**Phase 1**: a YOLOv8-based computer-vision model that detects high-visibility
vest compliance in airport ramp and airside environments.

**Detection classes (v1):**
- `vest` — person wearing a high-visibility vest
- `no-vest` — person NOT wearing a high-visibility vest (violation)

This module is the foundation for the broader ASSIS roadmap: FOD detection,
slip/trip/fall identification, badge-misuse alerting, and SIDA-area anomaly
detection — aligned with FAA Part 139 Safety Management Systems and TSA Part
1542 security frameworks.

## Why This Matters

- U.S. airports are designated critical infrastructure (DHS/CISA)
- Ramp operations carry persistent injury and ground-damage risk; industry
  estimates place ground-damage costs in the billions annually
- PPE compliance monitoring today is manual, intermittent, and reactive
- Small and non-hub airports lack resources for commercial safety-analytics
  platforms — a gap identified in ACRP research on SMS implementation

## Project Structure

```
ASSIS-PPE-Detection/
├── README.md
├── LICENSE                    ← MIT
├── requirements.txt
├── notebooks/
│   └── ASSIS_PPE_Training_Colab.ipynb   ← Train the model (Google Colab)
├── src/
│   ├── train.py               ← Training script
│   ├── detect.py              ← CLI detection on images/video
│   └── utils.py               ← Compliance scoring logic
├── app/
│   └── streamlit_app.py       ← Interactive demo web app
├── docs/
│   ├── DATASET_GUIDE.md       ← Getting and preparing training data
│   ├── WHITE_PAPER_OUTLINE.md ← Research paper skeleton
│   ├── PROJECT_ROADMAP.md     ← 90-day execution plan
│   └── RESEARCH_LOG.md        ← Dated progress log
├── data/                      ← Datasets (gitignored)
└── models/                    ← Trained weights (gitignored)
```

## Quickstart

### 1. Get the dataset
Follow `docs/DATASET_GUIDE.md` — download a PPE dataset from Roboflow
Universe (free) in YOLOv8 format.

### 2. Train
Open `notebooks/ASSIS_PPE_Training_Colab.ipynb` in Google Colab, enable a
T4 GPU runtime, run top to bottom (~1–3 hours).

### 3. Demo
Place your trained `best.pt` in `models/`, then:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Upload a ramp photo → bounding boxes + live compliance dashboard.

## Tech Stack

| Component | Tool |
|---|---|
| Object detection | YOLOv8n (Ultralytics) |
| Training | Google Colab (T4 GPU) |
| Dataset management | Roboflow |
| Demo interface | Streamlit |

## Regulatory Alignment

Supports compliance monitoring consistent with:
- **FAA 14 CFR Part 139** — Airport certification & SMS
- **TSA 49 CFR Part 1542** — Airport security programs
- **ICAO Annex 19** — Safety risk management
- **OSHA 29 CFR 1910.132** — PPE requirements

ASSIS is a research framework and decision-support layer. It does not replace
regulatory obligations, human judgment, or existing safety programs. No
operational CCTV or SSI material is used in this repository.

## Roadmap

- **Phase 1 (this repo):** PPE vest detection
- **Phase 2:** FOD detection on aprons/taxiways
- **Phase 3:** Slip/trip/fall pose estimation
- **Phase 4:** Multi-module integration + SMS data pipeline

See `docs/PROJECT_ROADMAP.md` and `docs/RESEARCH_LOG.md`.

## Author & Context

Developed as independent research in aviation safety technology, informed by
operational experience at Denver International Airport (DEN), Dallas Love
Field (DAL), and Charleston International Airport (CHS), and research
engagement with TRB ACRP Project 04-26 (Safety Management Systems for Small
and Non-Hub Airports).

## License

MIT — free to use, modify, and distribute with attribution.
