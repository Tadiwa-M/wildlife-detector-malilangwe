"""
Evaluate a trained YOLOv11 model on the WAID dataset.

Usage:
    python scripts/evaluate.py --weights weights/best.pt
    python scripts/evaluate.py --weights weights/best.pt --split test
    python scripts/evaluate.py --weights weights/best.pt --save-plots
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

from src.config import load_config
from src.data.dataset import generate_dataset_yaml
from src.utils.logging_setup import setup_logging

import logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained wildlife detector on the WAID dataset"
    )
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to trained model weights (.pt file).",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a YAML config override file.",
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["val", "test"],
        help="Dataset split to evaluate on (default: test).",
    )
    parser.add_argument(
        "--save-plots", action="store_true",
        help="Save confusion matrix and PR curve plots.",
    )
    parser.add_argument(
        "--conf", type=float, default=None,
        help="Override confidence threshold for evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg)

    weights_path = Path(args.weights)
    if not weights_path.exists():
        logger.error("Weights file not found: %s", weights_path)
        sys.exit(1)

    # Generate dataset YAML
    dataset_yaml = generate_dataset_yaml(cfg)

    # Load model
    logger.info("Loading model from %s", weights_path)
    model = YOLO(str(weights_path))

    # Run validation
    det_cfg = cfg.detection
    conf = args.conf if args.conf is not None else float(det_cfg.confidence_threshold)

    logger.info("Evaluating on '%s' split with conf=%.2f", args.split, conf)
    metrics = model.val(
        data=str(dataset_yaml),
        split=args.split,
        conf=conf,
        iou=float(det_cfg.iou_threshold),
        imgsz=int(det_cfg.image_size),
        device=str(det_cfg.device),
        plots=args.save_plots,
    )

    # Print results
    print("\n" + "=" * 60)
    print(f"  Evaluation Results — {args.split} split")
    print("=" * 60)
    print(f"  mAP@50:      {metrics.box.map50:.4f}")
    print(f"  mAP@50-95:   {metrics.box.map:.4f}")
    print(f"  Precision:    {metrics.box.mp:.4f}")
    print(f"  Recall:       {metrics.box.mr:.4f}")
    print("=" * 60)

    # Per-class results
    class_names = model.names
    if hasattr(metrics.box, "ap50") and metrics.box.ap50 is not None:
        print("\n  Per-class AP@50:")
        for i, ap in enumerate(metrics.box.ap50):
            name = class_names.get(i, f"class_{i}")
            print(f"    {name:<12s} {ap:.4f}")

    # Save metrics to JSON
    results_dir = Path(str(cfg.paths.output_dir)) / "evaluation"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"metrics_{args.split}.json"

    results = {
        "split": args.split,
        "weights": str(weights_path),
        "confidence_threshold": conf,
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Metrics saved to %s", results_file)


if __name__ == "__main__":
    main()
