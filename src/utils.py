"""
ASSIS PPE Detection - shared utilities (v3).

Adds TRUE violation detection: correlates person detections (from the base
YOLOv8 COCO-pretrained model) with PPE-item detections (from the
fine-tuned ASSIS model) to determine which people are NOT wearing required
PPE - not just whether PPE items are present somewhere in the frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

PPE_CLASS_NAMES = {
    0: "gloves",
    1: "helmet",
    2: "vest",
}

COCO_PERSON_CLASS_ID = 0
OVERLAP_THRESHOLD = 0.10


def _box_overlap_fraction(person_box, item_box) -> float:
    px1, py1, px2, py2 = person_box
    ix1, iy1, ix2, iy2 = item_box
    inter_x1 = max(px1, ix1)
    inter_y1 = max(py1, iy1)
    inter_x2 = min(px2, ix2)
    inter_y2 = min(py2, iy2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    person_area = max(1e-6, (px2 - px1) * (py2 - py1))
    return inter_area / person_area


@dataclass
class PersonCompliance:
    person_bbox: list
    has_vest: bool
    has_helmet: bool
    has_gloves: bool
    confidence: float


@dataclass
class ComplianceSummary:
    people: list = field(default_factory=list)
    raw_ppe_counts: dict = field(default_factory=dict)

    @property
    def total_people(self) -> int:
        return len(self.people)

    @property
    def vest_compliant(self) -> int:
        return sum(1 for p in self.people if p.has_vest)

    @property
    def violations(self) -> int:
        return self.total_people - self.vest_compliant

    @property
    def compliance_rate(self) -> float:
        if self.total_people == 0:
            return 100.0
        return round(100.0 * self.vest_compliant / self.total_people, 1)

    @property
    def status(self) -> str:
        if self.total_people == 0:
            return "NO PERSONNEL DETECTED"
        if self.violations == 0:
            return "COMPLIANT - all personnel wearing required vest"
        return f"VIOLATION(S) DETECTED: {self.violations} - vest not detected"


def summarize_results(person_results, ppe_results) -> ComplianceSummary:
    summary = ComplianceSummary()
    ppe_boxes = {"vest": [], "helmet": [], "gloves": []}
    if ppe_results.boxes is not None:
        for box in ppe_results.boxes:
            cls_id = int(box.cls[0])
            label = PPE_CLASS_NAMES.get(cls_id)
            if label:
                xyxy = box.xyxy[0].tolist()
                ppe_boxes[label].append(xyxy)
                summary.raw_ppe_counts[label] = summary.raw_ppe_counts.get(label, 0) + 1

    if person_results.boxes is not None:
        for box in person_results.boxes:
            cls_id = int(box.cls[0])
            if cls_id != COCO_PERSON_CLASS_ID:
                continue
            person_box = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            has_vest = any(_box_overlap_fraction(person_box, v) >= OVERLAP_THRESHOLD for v in ppe_boxes["vest"])
            has_helmet = any(_box_overlap_fraction(person_box, h) >= OVERLAP_THRESHOLD for h in ppe_boxes["helmet"])
            has_gloves = any(_box_overlap_fraction(person_box, g) >= OVERLAP_THRESHOLD for g in ppe_boxes["gloves"])
            summary.people.append(PersonCompliance(
                person_bbox=[round(v, 1) for v in person_box],
                has_vest=has_vest, has_helmet=has_helmet, has_gloves=has_gloves,
                confidence=round(conf, 3),
            ))
    return summary
