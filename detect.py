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

    model = YOLO(str(weights))
    results = model(args.source, conf=args.conf, save=args.save,
                    project="runs", name="detect")

    for i, r in enumerate(results):
        s = summarize_results(r)
        print(f"\n--- Frame/Image {i + 1} ---")
        print(f"Personnel detected: {s.total_people}")
        print(f"Compliant (vest):   {s.compliant}")
        print(f"Violations:         {s.violations}")
        print(f"Compliance rate:    {s.compliance_rate}%")
        print(f"Status:             {s.status}")

    if args.save:
        print("\nAnnotated output saved under runs/detect/")


if __name__ == "__main__":
    main()
