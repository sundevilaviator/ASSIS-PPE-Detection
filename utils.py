"""ASSIS PPE Detection - shared utilities for compliance scoring."""

from __future__ import annotations

from dataclasses import dataclass, field

# Class names must match the dataset's data.yaml ordering.
CLASS_NAMES = {
    0: "vest",
    1: "no-vest",
}

CLASS_COLORS = {
    "vest": (0, 200, 0),
    "no-vest": (0, 0, 230),
}


@dataclass
class ComplianceSummary:
    """Aggregate stats for one image or video frame."""

    total_people: int = 0
    compliant: int = 0
    violations: int = 0
    detections: list = field(default_factory=list)

    @property
    def compliance_rate(self) -> float:
        if self.total_people == 0:
            return 100.0
        return round(100.0 * self.compliant / self.total_people, 1)

    @property
    def status(self) -> str:
        if self.total_people == 0:
            return "NO PERSONNEL DETECTED"
        if self.violations == 0:
            return "COMPLIANT"
        return f"{self.violations} VIOLATION(S) DETECTED"


def summarize_results(results) -> ComplianceSummary:
    """Convert Ultralytics YOLO results for one image into a ComplianceSummary."""
    summary = ComplianceSummary()

    if results.boxes is None:
        return summary

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        xyxy = [round(float(v), 1) for v in box.xyxy[0].tolist()]

        summary.total_people += 1
        if label == "vest":
            summary.compliant += 1
        else:
            summary.violations += 1

        summary.detections.append(
            {"label": label, "confidence": round(conf, 3), "bbox": xyxy}
        )

    return summary
