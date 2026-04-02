"""
Fine-tune YOLOv11 on the WAID dataset for aerial wildlife detection.

Usage:
    python scripts/train.py
    python scripts/train.py --config config/custom.yaml
    python scripts/train.py --resume runs/train/weights/last.pt
    python scripts/train.py --validate-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

from src.config import load_config
from src.data.dataset import generate_dataset_yaml, get_class_distribution, validate_dataset
from src.utils.logging_setup import setup_logging

import logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv11 on the WAID aerial wildlife dataset"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a YAML config override file.",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a checkpoint to resume training from.",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Only validate the dataset and print statistics, don't train.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg)

    # Validate dataset
    logger.info("Validating WAID dataset...")
    stats = validate_dataset(cfg)

    if stats["total_images"] == 0:
        logger.error(
            "No images found in dataset. Download WAID images first:\n"
            "  git clone https://github.com/xiaohuicui/WAID.git"
        )
        sys.exit(1)

    # Show class distribution
    logger.info("Class distribution (train split):")
    dist = get_class_distribution(cfg, split="train")
    for cls_name, count in sorted(dist.items(), key=lambda x: -x[1]):
        logger.info("  %-10s %6d", cls_name, count)

    if args.validate_only:
        logger.info("Validation complete. Exiting (--validate-only).")
        return

    # Generate dataset YAML for Ultralytics
    dataset_yaml = generate_dataset_yaml(cfg)
    logger.info("Dataset YAML: %s", dataset_yaml)

    # Resolve model
    train_cfg = cfg.training
    det_cfg = cfg.detection
    resume_path = args.resume

    if resume_path:
        logger.info("Resuming training from checkpoint: %s", resume_path)
        model = YOLO(resume_path)
    else:
        model_variant = str(det_cfg.model_variant)
        logger.info("Starting training with pretrained %s", model_variant)
        model = YOLO(f"{model_variant}.pt")

    # Train
    model.train(
        data=str(dataset_yaml),
        epochs=int(train_cfg.epochs),
        batch=int(train_cfg.batch_size),
        imgsz=int(train_cfg.image_size),
        optimizer=str(train_cfg.optimizer),
        lr0=float(train_cfg.learning_rate),
        weight_decay=float(train_cfg.weight_decay),
        patience=int(train_cfg.patience),
        device=str(det_cfg.device),
        # Augmentation
        hsv_h=float(train_cfg.augmentation.hsv_h),
        hsv_s=float(train_cfg.augmentation.hsv_s),
        hsv_v=float(train_cfg.augmentation.hsv_v),
        flipud=float(train_cfg.augmentation.flipud),
        fliplr=float(train_cfg.augmentation.fliplr),
        mosaic=float(train_cfg.augmentation.mosaic),
        mixup=float(train_cfg.augmentation.mixup),
        scale=float(train_cfg.augmentation.scale),
        resume=bool(resume_path),
    )

    logger.info("Training complete! Check runs/detect/train/ for results.")


if __name__ == "__main__":
    main()
