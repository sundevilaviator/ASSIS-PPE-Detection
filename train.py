"""ASSIS PPE Detection - training script.

Usage:
    python src/train.py --data data/ppe-dataset/data.yaml --epochs 50
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ASSIS PPE detector")
    parser.add_argument("--data", default="data/ppe-dataset/data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model", default="yolov8n.pt")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="runs",
        name="assis_ppe",
        patience=15,
        plots=True,
    )

    metrics = model.val()
    print("\n=== Validation Metrics ===")
    print(f"mAP50:     {metrics.box.map50:.3f}")
    print(f"mAP50-95:  {metrics.box.map:.3f}")
    print("\nBest weights: runs/assis_ppe/weights/best.pt -> copy to models/")


if __name__ == "__main__":
    main()
