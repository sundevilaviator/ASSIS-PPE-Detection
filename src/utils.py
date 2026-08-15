"""
ASSIS PPE Detection - shared utilities (v4).

Correlates person detections (COCO-pretrained base model) with PPE-item
detections (fine-tuned ASSIS model) to determine which people are NOT
wearing required PPE.

v4 adds three-state per-item status (see docs/PPE_TAXONOMY.md section 4.4):
    COMPLIANT      - required item observed
    VIOLATION      - required item confidently absent
    INDETERMINATE  - item not observed, but this class is not reliable
                     enough (or not required under current conditions)
                     for absence to be treated as a violation

Only `vest` currently has demonstrated cross-domain reliability (see
docs/RESEARCH_LOG.md, 2026-08-13 entries) and is the sole class gating the
overall compliance decision. `helmet`/hearing-protection and `gloves` are
reported for transparency but never produce a VIOLATION on their own until
promoted per the criterion in docs/PPE_TAXONOMY.md section 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

PPE_CLASS_NAMES = {
    0: "gloves",
    1: "helmet",  # NOTE: mislabelled for aviation use - see PPE_TAXONOMY.md.
    #        Currently detects head-mounted equipment generally
    #        (caps, hearing protection), not rigid hard hats.
    2: "vest",
}

# Classes with demonstrated cross-domain (aviation) reliability, allowed to
# produce a VIOLATION state. Update only after the promotion criterion in
# docs/PPE_TAXONOMY.md section 5 is met and recorded in RESEARCH_LOG.md.
GATING_CLASSES = {"vest"}

COCO_PERSON_CLASS_ID = 0
OVERLAP_THRESHOLD = 0.10

# Minimum person bounding-box height as a fraction of frame height. Detections
# smaller than this are excluded from compliance scoring entirely (not even
# INDETERMINATE - simply not counted as "personnel").
#
# CALIBRATION HISTORY (see RESEARCH_LOG.md for dated entries):
# - Initially set to 0.15, intended to exclude distant background people
#   (e.g. boarding passengers) from a crowded wide-shot test that produced
#   89 false violations.
# - That value was too aggressive: tested against a normal wide apron
#   establishing shot with real, legitimately-distant ramp agents (~6-8% of
#   frame height at typical working camera distance), it excluded EVERYONE,
#   including actual vested workers, producing 0 personnel detected on a
#   video with 4 real agents present. This was a worse failure than the
#   problem it was meant to fix.
# - Lowered to 0.03 to only exclude genuinely tiny/far-background
#   detections (noise-level, not real working-distance personnel), rather
#   than trying to separate "worker" from "passenger" by size - the two can
#   appear at similar apparent size (e.g. a passenger boarding stairs near
#   the aircraft vs. a ramp agent standing nearby), so size alone is a weak
#   and now demonstrably unreliable proxy for role.
#
# HONEST LIMITATION: this filter cannot reliably solve the original crowded-
# scene problem (distinguishing workers from bystanders) - it can only strip
# clear background noise. The real fix is still a configurable region-of-
# interest / exclusion-zone system (see docs/FOD_PHASE2_PLAN.md Section 4.3,
# "Exclusion zones" - the same concept applies directly here) or a person
# role classifier, neither of which exists yet.
MIN_PERSON_HEIGHT_FRACTION = 0.03


class ItemStatus(str, Enum):
    COMPLIANT = "COMPLIANT"
    VIOLATION = "VIOLATION"
    INDETERMINATE = "INDETERMINATE"


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


def _item_status(detected: bool, required: bool, gating: bool) -> ItemStatus:
    if detected:
        return ItemStatus.COMPLIANT
    if required and gating:
        return ItemStatus.VIOLATION
    return ItemStatus.INDETERMINATE


@dataclass
class PersonCompliance:
    person_bbox: list
    confidence: float
    detected: dict  # class -> bool, what was actually observed
    status: dict  # class -> ItemStatus, after applying requirements + gating

    @property
    def has_vest(self) -> bool:
        return self.detected.get("vest", False)

    @property
    def has_helmet(self) -> bool:
        return self.detected.get("helmet", False)

    @property
    def has_gloves(self) -> bool:
        return self.detected.get("gloves", False)

    @property
    def is_violation(self) -> bool:
        return any(s == ItemStatus.VIOLATION for s in self.status.values())


@dataclass
class ComplianceSummary:
    people: list = field(default_factory=list)
    raw_ppe_counts: dict = field(default_factory=dict)
    requirements_reasons: dict = field(default_factory=dict)
    excluded_small: int = 0

    @property
    def total_people(self) -> int:
        return len(self.people)

    @property
    def vest_compliant(self) -> int:
        return sum(1 for p in self.people if p.has_vest)

    @property
    def violations(self) -> int:
        return sum(1 for p in self.people if p.is_violation)

    @property
    def compliance_rate(self) -> float:
        if self.total_people == 0:
            return 100.0
        return round(100.0 * (self.total_people - self.violations) / self.total_people, 1)

    @property
    def status(self) -> str:
        if self.total_people == 0:
            return "NO PERSONNEL DETECTED"
        if self.violations == 0:
            return "COMPLIANT - all personnel meet gating PPE requirements"
        return f"VIOLATION(S) DETECTED: {self.violations}"


def summarize_results(person_results, ppe_results, requirements=None, frame_height=None) -> ComplianceSummary:
    """Build a compliance summary.

    `requirements` is an optional conditions.PPERequirements (see
    src/conditions.py). If omitted, every class is treated as required,
    which reproduces v3 behaviour for vest (the only gating class) and
    keeps helmet/gloves informational only.

    `frame_height` (pixels) enables the minimum-person-size filter (see
    MIN_PERSON_HEIGHT_FRACTION above). If omitted, no size filtering is
    applied - every detected person is scored regardless of distance,
    reproducing prior behaviour. Pass it explicitly for crowded/wide-shot
    scenes where distant background people should not count as "personnel."
    """
    summary = ComplianceSummary()
    if requirements is not None:
        summary.requirements_reasons = requirements.reasons

    ppe_boxes = {"vest": [], "helmet": [], "gloves": []}
    if ppe_results.boxes is not None:
        for box in ppe_results.boxes:
            cls_id = int(box.cls[0])
            label = PPE_CLASS_NAMES.get(cls_id)
            if label:
                xyxy = box.xyxy[0].tolist()
                ppe_boxes[label].append(xyxy)
                summary.raw_ppe_counts[label] = summary.raw_ppe_counts.get(label, 0) + 1

    req_map = {
        "vest": True,
        "helmet": getattr(requirements, "hearing_protection", True),
        "gloves": getattr(requirements, "gloves", False),
    }

    min_height_px = (
        frame_height * MIN_PERSON_HEIGHT_FRACTION if frame_height else 0.0
    )
    excluded_small = 0

    if person_results.boxes is not None:
        for box in person_results.boxes:
            cls_id = int(box.cls[0])
            if cls_id != COCO_PERSON_CLASS_ID:
                continue
            person_box = box.xyxy[0].tolist()
            conf = float(box.conf[0])

            box_height = person_box[3] - person_box[1]
            if box_height < min_height_px:
                excluded_small += 1
                continue

            detected = {
                label: any(
                    _box_overlap_fraction(person_box, item_box) >= OVERLAP_THRESHOLD
                    for item_box in boxes
                )
                for label, boxes in ppe_boxes.items()
            }
            item_status = {
                label: _item_status(
                    detected[label], req_map[label], label in GATING_CLASSES
                )
                for label in ppe_boxes
            }

            summary.people.append(
                PersonCompliance(
                    person_bbox=[round(v, 1) for v in person_box],
                    confidence=round(conf, 3),
                    detected=detected,
                    status=item_status,
                )
            )
    summary.excluded_small = excluded_small
    return summary
