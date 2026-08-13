"""ASSIS PPE Detection - command-line inference.

Usage:
    python src/detect.py --source path/to/image.jpg
    python src/detect.py --source folder/ --save
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from utils import summarize_results

BASE_WEIGHTS = "yolov8n.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ASSIS PPE detection")
    parser.add_argument("--source", required=True)
    parser.add_argument("--weights", default="models/best.pt")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(
            f"Weights not found at {weights}. Train first (see notebooks/) "
            "and copy best.pt into models/."
        )

    # Same dual-model design as the Streamlit app: a COCO person detector
    # supplies the personnel boxes, the fine-tuned model supplies PPE items,
    # and compliance is decided by overlap between the two.
    base_model = YOLO(BASE_WEIGHTS)
    ppe_model = YOLO(str(weights))

    person_batch = base_model(args.source, conf=args.conf, classes=[0])
    ppe_batch = ppe_model(
        args.source, conf=args.conf, save=args.save, project="runs", name="detect"
    )

    for i, (person_results, ppe_results) in enumerate(zip(person_batch, ppe_batch)):
        s = summarize_results(person_results, ppe_results)
        print(f"\n--- Frame/Image {i + 1} ---")
        print(f"Personnel detected: {s.total_people}")
        print(f"Compliant (vest):   {s.vest_compliant}")
        print(f"Violations:         {s.violations}")
        print(f"Compliance rate:    {s.compliance_rate}%")
        print(f"Status:             {s.status}")
        if s.raw_ppe_counts:
            print(f"Raw PPE counts:     {s.raw_ppe_counts}")

    if args.save:
        print("\nAnnotated output saved under runs/detect/")


if __name__ == "__main__":
    main()
