"""
Fine-tune YOLOv11 on WAID or a merged multi-dataset for aerial wildlife detection.

Usage:
    # WAID training (Phase A):
    python scripts/train.py
    python scripts/train.py --resume runs/train/weights/last.pt
    python scripts/train.py --validate-only

    # Merged dataset training (Phase A+):
    python scripts/train.py --dataset data/merged.yaml --base-weights weights/best.pt
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
        description="Fine-tune YOLOv11 on an aerial wildlife dataset"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a YAML config override file.",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Path to dataset YAML (overrides WAID default). "
             "Use data/merged.yaml for Phase A+ multi-dataset training.",
    )
    parser.add_argument(
        "--base-weights", type=str, default=None,
        help="Path to base model weights for transfer learning. "
             "Use weights/best.pt when training on merged dataset (Phase A+).",
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

    train_cfg = cfg.training
    det_cfg = cfg.detection
    is_merged = args.dataset is not None

    # ── Dataset YAML ─────────────────────────────────────────────────────────
    if is_merged:
        dataset_yaml = Path(args.dataset)
        if not dataset_yaml.exists():
            logger.error(
                "Dataset YAML not found: %s\n"
                "Run scripts/merge_datasets.py first.", dataset_yaml
            )
            sys.exit(1)
        logger.info("Using merged dataset: %s", dataset_yaml)
    else:
        logger.info("Validating WAID dataset...")
        stats = validate_dataset(cfg)
        if stats["total_images"] == 0:
            logger.error(
                "No images found. Download WAID images first:\n"
                "  git clone https://github.com/xiaohuicui/WAID.git"
            )
            sys.exit(1)
        logger.info("Class distribution (train split):")
        dist = get_class_distribution(cfg, split="train")
        for cls_name, count in sorted(dist.items(), key=lambda x: -x[1]):
            logger.info("  %-10s %6d", cls_name, count)

        if args.validate_only:
            logger.info("Validation complete. Exiting (--validate-only).")
            return

        dataset_yaml = generate_dataset_yaml(cfg)
        logger.info("Dataset YAML: %s", dataset_yaml)

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.resume:
        logger.info("Resuming from checkpoint: %s", args.resume)
        model = YOLO(args.resume)
        resume_flag = True
    elif args.base_weights:
        base = Path(args.base_weights)
        if not base.exists():
            logger.error("Base weights not found: %s", base)
            sys.exit(1)
        logger.info("Transfer learning from: %s", base)
        model = YOLO(str(base))
        resume_flag = False
    else:
        model_variant = str(det_cfg.model_variant)
        logger.info("Starting fresh from pretrained %s", model_variant)
        model = YOLO(f"{model_variant}.pt")
        resume_flag = False

    # ── Hyperparameters for merged dataset training ───────────────────────────
    # When doing transfer learning on merged dataset, use lower LR and fewer
    # epochs since we're fine-tuning an already-trained model.
    if is_merged and not args.resume:
        epochs = 30
        lr = 0.0005         # lower LR for transfer learning
        freeze = 10         # freeze first 10 backbone layers
        run_name = "merged_transfer"
        logger.info(
            "Phase A+ transfer learning: epochs=%d, lr=%.4f, freeze=%d layers",
            epochs, lr, freeze,
        )
    else:
        epochs = int(train_cfg.epochs)
        lr = float(train_cfg.learning_rate)
        freeze = None
        run_name = "waid_yolo11n"

    # ── Train ─────────────────────────────────────────────────────────────────
    train_kwargs = dict(
        data=str(dataset_yaml),
        epochs=epochs,
        batch=int(train_cfg.batch_size),
        imgsz=int(train_cfg.image_size),
        optimizer=str(train_cfg.optimizer),
        lr0=lr,
        weight_decay=float(train_cfg.weight_decay),
        patience=int(train_cfg.patience),
        device=str(det_cfg.device),
        hsv_h=float(train_cfg.augmentation.hsv_h),
        hsv_s=float(train_cfg.augmentation.hsv_s),
        hsv_v=float(train_cfg.augmentation.hsv_v),
        flipud=float(train_cfg.augmentation.flipud),
        fliplr=float(train_cfg.augmentation.fliplr),
        mosaic=float(train_cfg.augmentation.mosaic),
        mixup=float(train_cfg.augmentation.mixup),
        scale=float(train_cfg.augmentation.scale),
        project="runs/train",
        name=run_name,
        exist_ok=True,
        resume=resume_flag,
    )
    if freeze is not None:
        train_kwargs["freeze"] = freeze

    model.train(**train_kwargs)
    logger.info("Training complete! Check runs/train/%s/ for results.", run_name)


if __name__ == "__main__":
    main()
