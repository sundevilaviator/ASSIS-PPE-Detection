[README.md](https://github.com/user-attachments/files/31117395/README.md)

# ASSIS — RampGuard: AI-Integrated Airport Safety and Security Intelligence System — Phase 1 (PPE Compliance Detection)

![Workflow Diagram](https://github.com/sundevilaviator/ASSIS-PPE-Detection/raw/main/workflow.svg)

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

Ramp and airside operations at U.S. airports are a persistent, measurable source of occupational risk — industry estimates put worldwide ramp-accident costs at roughly $10 billion a year. High-visibility vests, head protection, and hand protection are OSHA-mandated controls for this risk (29 CFR §1910.132), but in practice, compliance monitoring is still mostly manual and intermittent.

ASSIS Phase 1 asks a narrower question: can a YOLOv8 model, fine-tuned on freely available construction-industry PPE imagery (not aviation-specific), be brought to a usable level of accuracy on aviation ramp footage, using only the CCTV infrastructure airports already have? Treated as a transfer-learning problem — the detection task itself is domain-general, but everything around it (SMS reporting integration, regulatory alignment, ramp-specific conditions) is domain-specific.

This is Phase 1 of a broader platform, expected to expand into Foreign Object Debris detection, fall/slip/trip identification, and credential-misuse alerting in later phases (see [Roadmap](#roadmap)).

---

## Dual-Model Architecture

Rather than training a dedicated "non-compliant person" class — labeled data for that is scarce — ASSIS determines compliance through correlation:

| Step | Component | Output |
|---|---|---|
| 1 | Base YOLOv8 model (COCO-pretrained, unmodified) | Person bounding boxes |
| 2 | Fine-tuned ASSIS model (gloves / helmet / vest) | PPE-item bounding boxes + confidence |
| 3 | Correlation engine | For each detected person, computes overlap fraction with each PPE-item box |
| 4 | Compliance classifier | Person is compliant if overlap ≥ threshold (τ = 0.10) for the required PPE category |

This produces a per-person compliance record (vest: yes/no, helmet: yes/no, gloves: yes/no) and an aggregate compliance rate, structured for direct integration with airport Safety Management System (SMS) reporting workflows. See `src/utils.py` for the implementation.

### Model Variant Selection

Ultralytics ships YOLOv8 in five sizes. We picked the nano variant (YOLOv8n) for the fine-tuned PPE detector based on the published trade-offs between model size and accuracy:

| Variant | Size (MB) | Baseline mAP (COCO, IoU 0.5:0.95) | Relative Inference Speed |
|---|---|---|---|
| YOLOv8n | 6.2 | 37.3 | Fastest |
| YOLOv8s | 21.5 | 44.9 | Fast |
| YOLOv8m | 49.7 | 50.2 | Moderate |
| YOLOv8l | 83.7 | 52.9 | Slower |
| YOLOv8x | 130.5 | 53.9 | Slowest |

For deployment on commodity CCTV hardware — especially at small-hub and non-hub airports without dedicated GPU servers — inference speed and ease of deployment matter more than the marginal accuracy gains from a bigger model. YOLOv8n's actual validation performance on the ASSIS PPE task (mAP50 = 0.922; see [Results](#results)) comes in well above its baseline COCO score, which makes sense given the fine-tuning task has far fewer classes and much more domain specificity than general 80-class object detection.

---

## Three-State Compliance & Conditional PPE Policy

Two changes came out of what deployment testing surfaced (see `docs/RESEARCH_LOG.md`, 2026-08-13 entries):

**Three states per item.** A simple compliant/violation split can't express when the detector itself isn't sure. Each PPE class now resolves to one of:

| State | Meaning |
|---|---|
| `COMPLIANT` | Required item observed |
| `VIOLATION` | Required item confidently absent — only classes with demonstrated cross-domain reliability can produce this state |
| `INDETERMINATE` | Item not observed, but this class (or condition) is not yet reliable enough to count as a violation |

Only `vest` currently gates the compliance decision. `helmet` and `gloves` are reported for transparency but are always `INDETERMINATE`, never `VIOLATION`, until they meet the promotion criterion documented in `docs/PPE_TAXONOMY.md`.

**Weather-conditional requirements, via live METAR.** Not every PPE class is needed in every condition — eye protection matters when it's windy or debris is blowing, gloves matter when it's cold, while a hi-vis vest and footwear are always required. `src/conditions.py` pulls live METAR for a given airport (NOAA Aviation Weather Center, no API key needed) and works out which classes are currently required, defaulting to the stricter set whenever conditions are unknown or the reading is more than 90 minutes old. Full policy spec in `docs/PPE_TAXONOMY.md` §4.

---

## Deployment Targets

| Target | Support | Notes |
|---|---|---|
| Existing airport CCTV + on-premises server (CPU or GPU) | Supported | Primary intended deployment context; no new capital hardware required |
| Cloud-hosted inference (Streamlit Community Cloud) | Supported | Current live demo deployment |
| Edge inference hardware (e.g., NVIDIA Jetson, Raspberry Pi + accelerator) | Planned | Target for Phase 2 field-validation, consistent with low-capital deployment objective for resource-constrained airports |
| Real-time video pipeline (persistent multi-frame tracking) | Planned | Video and live-camera modes now exist (interval-sampled, not continuous), but persistent cross-frame person tracking is not yet implemented — see Known Limitations |

The model's small footprint and fast per-frame inference (T4 GPU, sub-5ms; see [Results](#results)) are meant to make this deployable on the kind of modest, airport-owned infrastructure that's realistic at small-hub, non-hub, and general aviation facilities, not something that requires dedicated high-performance computing.

---

## Prerequisites

The codebase was developed and tested in Google Colab (T4 GPU) and is deployable via Streamlit Community Cloud.

### Core Detection Pipeline

```
python              3.10+
ultralytics          8.4.118 (pinned)
torch                2.11.x
opencv-python-headless  # also powers video sampling and local live-camera capture (app/streamlit_app.py)
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

**Cross-domain testing:** we ran the trained model against held-out aviation ramp photographs never seen in training or validation. Vest detections have consistently scored 0.88–0.92 confidence across several independent test images, correctly matched to the right person all the way through the pipeline (local, Colab, and the deployed app all agree — see `docs/RESEARCH_LOG.md`, 2026-08-13). That supports the construction-to-aviation transfer premise for vest specifically.

**This isn't a reportable accuracy number yet, and we want to be clear about that.** The figures above are single-image confidence scores on a handful of held-out photos, not precision/recall against a labeled aviation test set. Treat them as evidence the approach is worth pursuing, not a validated accuracy rate. An annotated aviation validation set (target: 100+ images) is planned before we'd make any accuracy claim in a peer-reviewed context (see [Roadmap](#roadmap)).

Full methodology and discussion are in the accompanying technical paper (see [Citation](#citation)).

---

## Known Limitations

Laid out plainly here, since getting this right matters more than sounding polished — this repository also supports evidence submitted to USCIS.

- **Gloves doesn't transfer to aviation imagery.** Zero detections on aviation test photos even at conf=0.01, despite scoring 0.804 mAP50 in-domain on the construction validation split. Likely cause: the object is small at typical apron camera distance, and there may not be enough in-domain examples. Threshold tuning doesn't fix this — it needs retraining on aviation-domain imagery at higher inference resolution (see `notebooks/ASSIS_PPE_Retrain_SmallObjects.py`).
- **Helmet is a taxonomy problem, not just a weak detector.** The class was trained on rigid construction hard hats, and ramp personnel don't wear those — they wear caps and hearing protection. Right now it fires on head-mounted equipment generally, which is really closer to hearing protection than a helmet. Re-scoping this is planned (see `docs/PPE_TAXONOMY.md`).
- **A confirmed false positive sits in the 0.40–0.55 confidence band** — a plain shirt in a non-ramp landside scene got misread as a vest at 0.44 confidence, against a genuine-detection baseline of 0.88–0.92. Raised the default threshold to 0.60 in response, and anything below 0.70 now shows as unverified in the UI. Documenting this rather than quietly fixing it and moving on, because it's evidence of real testing, not something to hide.
- **Eye protection and footwear aren't implemented as detection classes yet.** No camera can verify their actual protective rating — impact resistance, steel toe — regardless of how good the model gets. See `docs/PPE_TAXONOMY.md` §2 for what optical detection can and can't confirm.
- **Video and live-camera modes exist and have been tested on real footage**, not just still photos. Uploaded video is sampled at a fixed interval and each sampled frame runs through the full detection pipeline; a local live-camera mode supports testing with a physically connected camera, though that only works when the app is running locally — a cloud deployment has no access to local hardware. This is interval-based batch processing of a file or feed, not continuous live CCTV ingestion — see `docs/FOD_PHASE2_PLAN.md` §4.1 for that distinction.
- **Inference resolution turned out to matter for vest detection too, not just helmet/gloves.** The app had been running inference at a silent default of 640px regardless of the actual image or video resolution. Testing on real 1920×1080 ramp footage showed this caused complete vest-detection failure at normal wide-shot or elevated-camera distance. Raising it to `imgsz=1280` recovered detection (0 confidence to 0.849 on the identical footage). This is now set explicitly everywhere inference happens.
- **Dense or crowded wide-shot scenes have a real detection-confidence ceiling — this isn't a setting we haven't found yet.** In one crowded apron scene, a vest that was clearly visible to a human eye only scored 0.451 confidence — inside the same unreliable 0.40–0.55 band from the false positive above, so lowering the threshold to catch it would just reopen that problem. We tested this from four different angles (inference resolution, minimum-person-size filtering, detection NMS, confidence threshold) and kept landing on the same answer. An actual fix needs different infrastructure — closer or higher-resolution camera coverage, tiled sub-region inference, or retraining specifically on dense small-object footage — not more configuration tuning. Full diagnostic history in `docs/RESEARCH_LOG.md`, 2026-08-13 entries.
- **Crowded scenes also can't tell who's actually a worker.** A generic person detector has no idea whether it's looking at a ramp worker or a passenger walking by. A minimum-size filter (`MIN_PERSON_HEIGHT_FRACTION` in `src/utils.py`) cuts out distant background noise but doesn't solve this — a real fix needs a region-of-interest/exclusion-zone system (the same idea already specified for FOD in `docs/FOD_PHASE2_PLAN.md` §4.3) or an actual role classifier.

---

## Regulatory Alignment

Ramp worker PPE requirements come mainly from OSHA general industry standards and carrier ground operations manuals, not from FAA airport-certification rules directly. Consistent with:

- **OSHA 29 CFR §1910.132/.95/.136/.138/.133** — PPE, hearing protection, footwear, hand protection, eye protection
- **FAA 14 CFR Part 139 Subpart E** — Safety Management Systems (airport certification; this is the SMS integration context, not where the PPE mandate comes from)
- **TSA 49 CFR Part 1542** — Airport security programs
- **ICAO Annex 19** — Safety risk management

`docs/PPE_TAXONOMY.md` has the full class-by-class regulatory basis and notes which citations still need to be checked against source text before anything gets filed.

ASSIS is a research framework and decision-support layer. It doesn't replace regulatory obligations, human judgment, or existing safety programs, and it can't verify PPE certification ratings — ANSI/ISEA retroreflective class, steel-toe rating — only whether an item is visibly present, which is reported as such. No airport CCTV footage, Sensitive Security Information, or SIDA-area imagery is used anywhere in this repository.

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

The current architecture uses rectangular bounding boxes for both person and PPE-item localization. Overlap between rectangles is only an approximation of whether someone's actually wearing an item, and it's sensitive to crowded scenes or partial occlusion, where a bystander's box can incidentally overlap a nearby worker's PPE detection. Instance segmentation — pixel-level masks instead of boxes — is something we're considering as a precision upgrade to the correlation engine above, and it's achievable inside the existing YOLOv8 setup (`yolov8n-seg` and similar segmentation checkpoints) without changing the underlying architecture or toolchain. Flagging this as a candidate for a later phase, not a limitation of what's already been built.

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
