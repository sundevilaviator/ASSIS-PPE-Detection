# ASSIS: AI-Integrated Airport Safety and Security Intelligence System — Phase 1 (PPE Compliance Detection)

![Workflow Diagram](https://github.com/sundevilaviator/ASSIS-PPE-Detection/raw/main/workflow.png)

---

ASSIS Phase 1 is a dual-model computer-vision system for automated Personal Protective Equipment (PPE) compliance detection in airport ramp environments. It combines a COCO-pretrained YOLOv8 person detector with a fine-tuned YOLOv8 PPE-item detector (gloves, helmet, vest), correlating the two via bounding-box overlap analysis to determine true per-person compliance — not merely whether PPE items are present somewhere in a frame. The model is trained via transfer learning from publicly available construction-industry imagery and validated on real aviation ramp photography.

## Table of Contents

- [Overview](#overview)
- [Dual-Model Architecture](#dual-model-architecture)
- [Prerequisites](#prerequisites)
- [Dataset](#dataset)
- [Reproducibility](#reproducibility)
- [Results](#results)
- [Regulatory Alignment](#regulatory-alignment)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [License](#license)

---

## Overview

U.S. airport ramp and airside operations present a persistent, quantifiable source of occupational risk, with industry estimates placing ramp-accident costs at $10 billion annually worldwide. Personal Protective Equipment compliance — principally high-visibility vests, head protection, and hand protection — is a foundational OSHA-mandated control (29 CFR §1910.132) for mitigating this risk, yet compliance monitoring today remains almost entirely manual and intermittent.

ASSIS Phase 1 addresses a narrow, tractable research question: can a YOLOv8 object-detection model, fine-tuned on freely available, non-aviation (construction-industry) PPE imagery, achieve deployable detection accuracy when evaluated against aviation ramp imagery, using only infrastructure airports already possess (existing CCTV)? This is treated as a transfer-learning problem — the underlying visual task is domain-general, while the deployment context (aviation ramp operations, SMS reporting integration, regulatory alignment) is domain-specific.

This module is Phase 1 of the broader ASSIS platform, which is designed to expand to Foreign Object Debris (FOD) detection, fall/slip/trip identification, and credential-misuse alerting in subsequent phases (see [Roadmap](#roadmap)).

---

## Dual-Model Architecture

Rather than training a dedicated "non-compliant person" class — for which labeled data is scarce — ASSIS determines compliance through a correlation approach:

| Step | Component | Output |
|---|---|---|
| 1 | Base YOLOv8 model (COCO-pretrained, unmodified) | Person bounding boxes |
| 2 | Fine-tuned ASSIS model (gloves / helmet / vest) | PPE-item bounding boxes + confidence |
| 3 | Correlation engine | For each detected person, computes overlap fraction with each PPE-item box |
| 4 | Compliance classifier | Person is compliant if overlap ≥ threshold (τ = 0.10) for the required PPE category |

This produces a per-person compliance record (vest: yes/no, helmet: yes/no, gloves: yes/no) and an aggregate compliance rate, structured for direct integration with airport Safety Management System (SMS) reporting workflows. See `src/utils.py` for the implementation.

---

## Prerequisites

The codebase was developed and tested in Google Colab (T4 GPU) and is deployable via Streamlit Community Cloud.

### Core Detection Pipeline

```
python              3.10+
ultralytics          8.4.x
torch                2.11.x
opencv-python-headless
numpy
pillow
```

### Interactive Demo

```
streamlit
```

### Dataset Management

```
roboflow
```

### Installation

```bash
git clone https://github.com/sundevilaviator/ASSIS-PPE-Detection.git
cd ASSIS-PPE-Detection

python -m venv venv_assis
source venv_assis/bin/activate

pip install -r requirements.txt
```

To confirm the environment is correctly configured, run:

```bash
python -c "from ultralytics import YOLO; print('Environment ready.')"
```

---

## Dataset

Training data was sourced from a publicly available PPE-detection dataset (Roboflow Universe), consisting of construction-industry ramp/worksite imagery. The original dataset included five annotated classes (`boots`, `gloves`, `helmet`, `human`, `vest`); the class set was reduced to three (`gloves`, `helmet`, `vest`) for the reported model, with the `human` class removed as redundant to the dedicated person-detection stage described above.

| Split | Images |
|---|---|
| Train | 1,566 |
| Validation | 420 |

See `docs/DATASET_GUIDE.md` for full dataset acquisition guidance and licensing notes.

---

## Reproducibility

### I. Environment Setup

Open `notebooks/ASSIS_PPE_Training_Colab.ipynb` in Google Colab. Set the hardware accelerator to a T4 GPU (`Runtime → Change runtime type → T4 GPU`) before executing any cells.

### II. Dataset Acquisition

Obtain a PPE-detection dataset from Roboflow Universe in YOLOv8 export format (see `docs/DATASET_GUIDE.md`). Insert your Roboflow API credentials into the designated notebook cell and execute to download the dataset into the Colab environment.

### III. Class Filtering (Optional — Reproduces the 3-Class Configuration)

To reproduce the reported 3-class configuration (gloves, helmet, vest) from a 5-class source dataset, run the label-remapping cell in the training notebook, which filters and re-indexes annotation files and updates `data.yaml` accordingly.

### IV. Model Training

Execute the training cell:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.train(
    data=data_yaml,
    epochs=50,
    imgsz=640,
    batch=16,
    patience=15,
    project="/content/drive/MyDrive/ASSIS_runs",
    name="assis_ppe",
)
```

Training completes in approximately 25 minutes on a single T4 GPU.

### V. Evaluation

```python
metrics = model.val()
print(f"mAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")
print(f"Precision: {metrics.box.mp:.3f}")
print(f"Recall: {metrics.box.mr:.3f}")
```

### VI. Local / Deployed Demo

```bash
streamlit run app/streamlit_app.py
```

Upload a ramp or worksite photograph to receive live person detection, PPE-item detection, and per-person compliance classification.

---

## Results

Validation set performance (3-class configuration: gloves, helmet, vest), reproduced across independent training runs:

| Metric | Value |
|---|---|
| mAP50 | 0.922 |
| mAP50-95 | 0.725 |
| Precision | 0.922 |
| Recall | 0.864 |

| Class | mAP50 | Validation Instances |
|---|---|---|
| vest | 0.985 | 467 |
| helmet | 0.978 | 320 |
| gloves | 0.804 | 41 |

**Cross-domain validation:** the trained model was evaluated on two independently sourced, previously unseen aviation ramp photographs (excluded from training and validation). Both were correctly classified for vest compliance (confidence 0.92 and 0.80 respectively), supporting the construction-to-aviation transfer-learning premise despite an entirely non-aviation training domain.

Full methodology, results, and discussion are reported in the accompanying technical paper (see [Citation](#citation)).

---

## Regulatory Alignment

Supports compliance monitoring consistent with:

- **FAA 14 CFR Part 139 Subpart E** — Safety Management Systems
- **TSA 49 CFR Part 1542** — Airport security programs
- **ICAO Annex 19** — Safety risk management
- **OSHA 29 CFR §1910.132** — PPE requirements

ASSIS is a research framework and decision-support layer. It does not replace regulatory obligations, human judgment, or existing safety programs. No airport CCTV footage, Sensitive Security Information, or SIDA-area imagery is used in this repository.

---

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| Phase 1 | PPE compliance detection (this repository) | Complete |
| Phase 2 | Foreign Object Debris (FOD) detection | In progress |
| Phase 3 | Slip/trip/fall pose-estimation detection | Planned |
| Phase 4 | Multi-module integration, credential-misuse alerting, unified SMS data pipeline | Planned |

See `docs/PROJECT_ROADMAP.md` and `docs/RESEARCH_LOG.md` for detailed, dated progress.

---

## Citation

If you use this code or reference this work, please cite:

```
Usman, S. ASSIS: AI-Integrated Airport Safety and Security Intelligence
System — Phase 1 (PPE Compliance Detection). 2026.
https://github.com/sundevilaviator/ASSIS-PPE-Detection
```

A machine-readable citation is also provided in `CITATION.cff`.

---

## License

Released under AGPL-3.0. See `LICENSE` and `NOTICE.md` for full details, including the project's forward-looking publication policy for subsequent development phases.
