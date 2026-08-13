<img width="200" height="150" alt="workflow" src="https://github.com/user-attachments/assets/8b8a767d-1b29-44b4-9255-62c4e0bf787a" />
# ASSIS — RampGuard: AI-Integrated Airport Safety and Security Intelligence System — Phase 1 (PPE Compliance Detection)

<img width="2080" height="1560" alt="image" src="https://github.com/user-attachments/assets/94e8d3cd-ddcf-4e3a-a507-aa5ec9726c82" />

---

ASSIS Phase 1, codenamed **RampGuard**, is a dual-model computer-vision system for automated Personal Protective Equipment (PPE) compliance detection in airport ramp environments. It combines a COCO-pretrained YOLOv8 person detector with a fine-tuned YOLOv8 PPE-item detector (gloves, helmet, vest), correlating the two via bounding-box overlap analysis to determine true per-person compliance — not merely whether PPE items are present somewhere in a frame. Compliance is reported in three states (compliant / violation / indeterminate) rather than a binary pass/fail, and PPE requirements are conditioned on live weather data (METAR) rather than fixed. The model is trained via transfer learning from publicly available construction-industry imagery and validated on real aviation ramp photography.

## Table of Contents

- [Overview](#overview)
- [Dual-Model Architecture](#dual-model-architecture)
- [Three-State Compliance & Conditional PPE Policy](#three-state-compliance--conditional-ppe-policy)
- [Deployment Targets](#deployment-targets)
- [Prerequisites](#prerequisites)
- [Dataset](#dataset)
- [Reproducibility](#reproducibility)
- [Results](#results)
- [Known Limitations](#known-limitations)
- [Regulatory Alignment](#regulatory-alignment)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [License](#license)

---

## Overview

U.S. airport ramp and airside operations present a persistent, quantifiable source of occupational risk, with industry estimates placing ramp-accident costs at $10 billion annually worldwide. Personal Protective Equipment compliance, principally high-visibility vests, head protection, and hand protection; is a foundational OSHA-mandated control (29 CFR §1910.132) for mitigating this risk, yet compliance monitoring today remains almost entirely manual and intermittent.

ASSIS Phase 1 addresses a narrow, tractable research question: can a YOLOv8 object-detection model, fine-tuned on freely available, non-aviation (construction-industry) PPE imagery, achieve deployable detection accuracy when evaluated against aviation ramp imagery, using only infrastructure airports already possess (existing CCTV)? This is treated as a transfer-learning problem. The underlying visual task is domain-general, while the deployment context (aviation ramp operations, SMS reporting integration, regulatory alignment) is domain-specific.

This module is Phase 1 of the broader ASSIS platform, which is designed to expand to Foreign Object Debris (FOD) detection, fall/slip/trip identification, and credential-misuse alerting in subsequent phases (see [Roadmap](#roadmap)).

---

## Dual-Model Architecture

Rather than training a dedicated "non-compliant person" class, for which labeled data is scarce — ASSIS determines compliance through a correlation approach:

| Step | Component | Output |
|---|---|---|
| 1 | Base YOLOv8 model (COCO-pretrained, unmodified) | Person bounding boxes |
| 2 | Fine-tuned ASSIS model (gloves / helmet / vest) | PPE-item bounding boxes + confidence |
| 3 | Correlation engine | For each detected person, computes overlap fraction with each PPE-item box |
| 4 | Compliance classifier | Person is compliant if overlap ≥ threshold (τ = 0.10) for the required PPE category |

This produces a per-person compliance record (vest: yes/no, helmet: yes/no, gloves: yes/no) and an aggregate compliance rate, structured for direct integration with airport Safety Management System (SMS) reporting workflows. See `src/utils.py` for the implementation.

### Model Variant Selection

Ultralytics distributes YOLOv8 in five parameter scales. The nano variant (YOLOv8n) was selected for the fine-tuned PPE detector on the basis of the following published trade-offs between model footprint and detection accuracy:

| Variant | Size (MB) | Baseline mAP (COCO, IoU 0.5:0.95) | Relative Inference Speed |
|---|---|---|---|
| YOLOv8n | 6.2 | 37.3 | Fastest |
| YOLOv8s | 21.5 | 44.9 | Fast |
| YOLOv8m | 49.7 | 50.2 | Moderate |
| YOLOv8l | 83.7 | 52.9 | Slower |
| YOLOv8x | 130.5 | 53.9 | Slowest |

For airport ramp deployment on commodity CCTV infrastructure, particularly at small-hub and non-hub facilities without dedicated GPU server hardware, inference latency and edge-deployability are weighted more heavily than the marginal accuracy gains offered by larger variants. YOLOv8n's validation performance on the ASSIS PPE task (mAP50 = 0.922; see [Results](#results)) substantially exceeds its baseline COCO accuracy, consistent with the reduced class complexity and domain specificity of the fine-tuning task relative to general-purpose 80-class object detection.

---

## Three-State Compliance & Conditional PPE Policy

Two refinements were made after initial deployment testing surfaced their necessity (see `docs/RESEARCH_LOG.md`, 2026-08-13 entries):

**Three-state per-item status.** Binary compliant/violation cannot express detector uncertainty. Each PPE class now resolves to one of:

| State | Meaning |
|---|---|
| `COMPLIANT` | Required item observed |
| `VIOLATION` | Required item confidently absent — only classes with demonstrated cross-domain reliability can produce this state |
| `INDETERMINATE` | Item not observed, but this class (or condition) is not yet reliable enough to count as a violation |

Only `vest` currently gates the compliance decision. `helmet` and `gloves` are reported for transparency but are always `INDETERMINATE`, never `VIOLATION`, until they meet the promotion criterion documented in `docs/PPE_TAXONOMY.md`.

**Conditional requirements via live METAR.** Not all PPE is required in all conditions — eye protection is situational (wind, blowing debris), gloves are situational (cold), while a hi-vis vest and footwear are unconditional. `src/conditions.py` fetches live METAR for a given airport (NOAA Aviation Weather Center, no API key required) and derives which classes are currently required, failing safe toward the stricter requirement whenever conditions are unknown or the observation is stale (>90 min). See `docs/PPE_TAXONOMY.md` §4 for the full policy specification.

---

| Target | Support | Notes |
|---|---|---|
| Existing airport CCTV + on-premises server (CPU or GPU) | Supported | Primary intended deployment context; no new capital hardware required |
| Cloud-hosted inference (Streamlit Community Cloud) | Supported | Current live demo deployment |
| Edge inference hardware (e.g., NVIDIA Jetson, Raspberry Pi + accelerator) | Planned | Target for Phase 2 field-validation, consistent with low-capital deployment objective for resource-constrained airports |
| Real-time video pipeline (persistent multi-frame tracking) | Planned | Current implementation evaluated on static frames; see Limitations in accompanying technical paper |

The system's low parameter count and sub-5ms per-frame inference latency (T4 GPU; see [Results](#results)) are intended to support deployment on modest, airport-owned infrastructure rather than requiring dedicated high-performance computing resources; a design constraint informed by the resource limitations documented at small-hub, non-hub, and general aviation facilities.

---

## Prerequisites

The codebase was developed and tested in Google Colab (T4 GPU) and is deployable via Streamlit Community Cloud.

### Core Detection Pipeline

```
python              3.10+
ultralytics          8.4.118 (pinned)
torch                2.11.x
opencv-python-headless
numpy
pillow
requests             # METAR fetching (src/conditions.py)
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

**Cross-domain validation:** the trained model was evaluated on held-out, previously unseen aviation ramp photographs (excluded from training and validation). `vest` detections have consistently scored 0.88–0.92 confidence across multiple independent test images, correctly correlated to personnel through the full pipeline (local, Colab, and deployed environments verified to agree — see `docs/RESEARCH_LOG.md`, 2026-08-13). This supports the construction-to-aviation transfer-learning premise for the `vest` class specifically.

**This is not yet a reportable accuracy metric.** The figures above are single-image confidence scores on a handful of held-out photographs, not precision/recall measured against a labeled aviation test set. Treat them as evidence the approach works, not as a validated accuracy rate — an annotated aviation validation set (target: 100+ images) is planned before any accuracy claim is made in a peer-reviewed context (see [Roadmap](#roadmap)).

Full methodology, results, and discussion are reported in the accompanying technical paper (see [Citation](#citation)).

---

## Known Limitations

Documented plainly, since precision here matters more than optimism (see project context: this repository supports evidence submitted to USCIS).

- **`gloves` does not transfer to aviation imagery.** Zero detections on aviation test photographs even at conf=0.01, despite 0.804 mAP50 in-domain (construction validation split). Root cause: small object scale at typical apron camera distance, and possibly insufficient in-domain examples. Not fixed by threshold tuning — requires retraining on aviation-domain imagery at higher inference resolution (see `notebooks/ASSIS_PPE_Retrain_SmallObjects.py`).
- **`helmet` is a taxonomy mismatch, not just a detection gap.** The class was trained on rigid construction hard hats. Ramp personnel do not wear hard hats; they wear caps and hearing protection. The class currently fires on head-mounted equipment generally, which is closer to hearing protection than a helmet — re-scoping is planned (see `docs/PPE_TAXONOMY.md`).
- **A confirmed false positive exists in the 0.40–0.55 confidence band** (a plain shirt in a non-ramp, landside scene misread as a vest at 0.44 confidence, against a genuine-detection baseline of 0.88–0.92). The application default threshold was raised to 0.60 in response, and detections below 0.70 are flagged as unverified in the UI. This is documented, not hidden, because it's evidence of active testing rigor rather than a reason to suppress the finding.
- **Eye protection and footwear are not yet implemented as detection classes.** Camera-based verification of their *protective rating* (impact resistance, steel toe) is not physically possible regardless of model accuracy — see `docs/PPE_TAXONOMY.md` §2 for what optical detection can and cannot verify.
- **Evaluated on static frames**, not video, and not under crowded/heavily-occluded conditions.

---

## Regulatory Alignment

Ramp worker PPE requirements derive principally from OSHA general industry standards and carrier ground operations manuals, not from FAA airport-certification rules directly. Supports compliance monitoring consistent with:

- **OSHA 29 CFR §1910.132/.95/.136/.138/.133** — PPE, hearing protection, footwear, hand protection, eye protection
- **FAA 14 CFR Part 139 Subpart E** — Safety Management Systems (airport certification; provides the SMS integration context, not the PPE mandate itself)
- **TSA 49 CFR Part 1542** — Airport security programs
- **ICAO Annex 19** — Safety risk management

See `docs/PPE_TAXONOMY.md` for the full class-by-class regulatory basis and a note on which citations require verification against source text before use in any filing.

ASSIS is a research framework and decision-support layer. It does not replace regulatory obligations, human judgment, or existing safety programs, and does not verify PPE certification ratings (e.g., ANSI/ISEA retroreflective class, steel-toe rating) — only the observable presence of an item, which is reported as such. No airport CCTV footage, Sensitive Security Information, or SIDA-area imagery is used in this repository.

---

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| Phase 1 | PPE vest compliance detection (this repository) | Vest validated; helmet/gloves characterized as open limitations, not resolved |
| Phase 1.5 | Aviation-domain retraining for gloves; hearing-protection re-scoping; annotated aviation validation set | In progress |
| Phase 2 | Foreign Object Debris (FOD) detection | Planned |
| Phase 3 | Slip/trip/fall pose-estimation detection | Planned |
| Phase 4 | Multi-module integration, credential-misuse alerting, unified SMS data pipeline | Planned |

### Under Consideration: Instance Segmentation

The current architecture performs bounding-box (rectangular) detection for both person and PPE-item localization. Bounding-box overlap is an approximation of physical PPE-wearing status and can be sensitive to crowded scenes or partial occlusion, where a bystander's bounding box may incidentally overlap a nearby worker's PPE detection. Instance segmentation, pixel-level object masks rather than rectangular boxes, is under consideration as a precision enhancement to the correlation engine described above, and is achievable within the existing YOLOv8 framework (`yolov8n-seg` and comparable segmentation-variant checkpoints) without a change of underlying architecture or toolchain. This is noted as a candidate refinement for a subsequent phase rather than a limitation of the present results.

See `docs/PROJECT_ROADMAP.md`, `docs/RESEARCH_LOG.md`, `docs/PPE_TAXONOMY.md`, and `docs/UI_UX_BLUEPRINT.md` for detailed, dated progress and design rationale.

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

---

## References

[1] Ultralytics. *YOLOv8 Documentation*. https://docs.ultralytics.com

[2] Federal Aviation Administration. (2023). *Safety Management Systems for Certificated Airports*, 14 CFR Part 139 Subpart E, Final Rule.

[3] Occupational Safety and Health Administration. 29 CFR §1910.132 — Personal Protective Equipment, General Requirements.

[4] Flight Safety Foundation. *Ground Accident Prevention (GAP) Program — "Covering the Ground."* AeroSafety World.

Full citation list, including regulatory and industry sources, is provided in the accompanying technical paper (see [Citation](#citation)).
[README.md](https://github.com/user-attachments/files/31045638/README.md)
# ASSIS: AI-Integrated Airport Safety and Security Intelligence System — Phase 1 (PPE Compliance Detection)

![Workflow Diagram](https://github.com/sundevilaviator/ASSIS-PPE-Detection/raw/main/workflow.png)

---

ASSIS Phase 1 is a dual-model computer-vision system for automated Personal Protective Equipment (PPE) compliance detection in airport ramp environments. It combines a COCO-pretrained YOLOv8 person detector with a fine-tuned YOLOv8 PPE-item detector (gloves, helmet, vest), correlating the two via bounding-box overlap analysis to determine true per-person compliance; not merely whether PPE items are present somewhere in a frame. The model is trained via transfer learning from publicly available construction-industry imagery and validated on real aviation ramp photography.

## Table of Contents

- [Overview](#overview)
- [Dual-Model Architecture](#dual-model-architecture)
- [Deployment Targets](#deployment-targets)
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

U.S. airport ramp and airside operations present a persistent, quantifiable source of occupational risk, with industry estimates placing ramp-accident costs at $10 billion annually worldwide. Personal Protective Equipment compliance, principally high-visibility vests, head protection, and hand protection; is a foundational OSHA-mandated control (29 CFR §1910.132) for mitigating this risk, yet compliance monitoring today remains almost entirely manual and intermittent.

ASSIS Phase 1 addresses a narrow, tractable research question: can a YOLOv8 object-detection model, fine-tuned on freely available, non-aviation (construction-industry) PPE imagery, achieve deployable detection accuracy when evaluated against aviation ramp imagery, using only infrastructure airports already possess (existing CCTV)? This is treated as a transfer-learning problem. The underlying visual task is domain-general, while the deployment context (aviation ramp operations, SMS reporting integration, regulatory alignment) is domain-specific.

This module is Phase 1 of the broader ASSIS platform, which is designed to expand to Foreign Object Debris (FOD) detection, fall/slip/trip identification, and credential-misuse alerting in subsequent phases (see [Roadmap](#roadmap)).

---

## Dual-Model Architecture

Rather than training a dedicated "non-compliant person" class, for which labeled data is scarce — ASSIS determines compliance through a correlation approach:

| Step | Component | Output |
|---|---|---|
| 1 | Base YOLOv8 model (COCO-pretrained, unmodified) | Person bounding boxes |
| 2 | Fine-tuned ASSIS model (gloves / helmet / vest) | PPE-item bounding boxes + confidence |
| 3 | Correlation engine | For each detected person, computes overlap fraction with each PPE-item box |
| 4 | Compliance classifier | Person is compliant if overlap ≥ threshold (τ = 0.10) for the required PPE category |

This produces a per-person compliance record (vest: yes/no, helmet: yes/no, gloves: yes/no) and an aggregate compliance rate, structured for direct integration with airport Safety Management System (SMS) reporting workflows. See `src/utils.py` for the implementation.

### Model Variant Selection

Ultralytics distributes YOLOv8 in five parameter scales. The nano variant (YOLOv8n) was selected for the fine-tuned PPE detector on the basis of the following published trade-offs between model footprint and detection accuracy:

| Variant | Size (MB) | Baseline mAP (COCO, IoU 0.5:0.95) | Relative Inference Speed |
|---|---|---|---|
| YOLOv8n | 6.2 | 37.3 | Fastest |
| YOLOv8s | 21.5 | 44.9 | Fast |
| YOLOv8m | 49.7 | 50.2 | Moderate |
| YOLOv8l | 83.7 | 52.9 | Slower |
| YOLOv8x | 130.5 | 53.9 | Slowest |

For airport ramp deployment on commodity CCTV infrastructure, particularly at small-hub and non-hub facilities without dedicated GPU server hardware, inference latency and edge-deployability are weighted more heavily than the marginal accuracy gains offered by larger variants. YOLOv8n's validation performance on the ASSIS PPE task (mAP50 = 0.922; see [Results](#results)) substantially exceeds its baseline COCO accuracy, consistent with the reduced class complexity and domain specificity of the fine-tuning task relative to general-purpose 80-class object detection.

---

## Deployment Targets

| Target | Support | Notes |
|---|---|---|
| Existing airport CCTV + on-premises server (CPU or GPU) | Supported | Primary intended deployment context; no new capital hardware required |
| Cloud-hosted inference (Streamlit Community Cloud) | Supported | Current live demo deployment |
| Edge inference hardware (e.g., NVIDIA Jetson, Raspberry Pi + accelerator) | Planned | Target for Phase 2 field-validation, consistent with low-capital deployment objective for resource-constrained airports |
| Real-time video pipeline (persistent multi-frame tracking) | Planned | Current implementation evaluated on static frames; see Limitations in accompanying technical paper |

The system's low parameter count and sub-5ms per-frame inference latency (T4 GPU; see [Results](#results)) are intended to support deployment on modest, airport-owned infrastructure rather than requiring dedicated high-performance computing resources; a design constraint informed by the resource limitations documented at small-hub, non-hub, and general aviation facilities.

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

### Under Consideration: Instance Segmentation

The current architecture performs bounding-box (rectangular) detection for both person and PPE-item localization. Bounding-box overlap is an approximation of physical PPE-wearing status and can be sensitive to crowded scenes or partial occlusion, where a bystander's bounding box may incidentally overlap a nearby worker's PPE detection. Instance segmentation, pixel-level object masks rather than rectangular boxes, is under consideration as a precision enhancement to the correlation engine described above, and is achievable within the existing YOLOv8 framework (`yolov8n-seg` and comparable segmentation-variant checkpoints) without a change of underlying architecture or toolchain. This is noted as a candidate refinement for a subsequent phase rather than a limitation of the present results.

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

---

## References

[1] Ultralytics. *YOLOv8 Documentation*. https://docs.ultralytics.com

[2] Federal Aviation Administration. (2023). *Safety Management Systems for Certificated Airports*, 14 CFR Part 139 Subpart E, Final Rule.

[3] Occupational Safety and Health Administration. 29 CFR §1910.132 — Personal Protective Equipment, General Requirements.

[4] Flight Safety Foundation. *Ground Accident Prevention (GAP) Program — "Covering the Ground."* AeroSafety World.

Full citation list, including regulatory and industry sources, is provided in the accompanying technical paper (see [Citation](#citation)).
