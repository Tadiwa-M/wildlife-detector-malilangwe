"""
Merge multiple aerial wildlife datasets into a single unified dataset
for Phase A+ training.

Usage:
    # Minimum — WAID only (dry run to test pipeline):
    python scripts/merge_datasets.py --waid WAID/WAID

    # WAID + AED (recommended minimum for elephant class):
    python scripts/merge_datasets.py --waid WAID/WAID --aed path/to/AED

    # Full merge:
    python scripts/merge_datasets.py \\
        --waid WAID/WAID \\
        --aed path/to/AED \\
        --liege path/to/liege \\
        --wildlifemapper path/to/wildlifemapper \\
        --mmla path/to/mmla

Output:
    data/merged/          — merged images and labels
    data/merged.yaml      — Ultralytics training YAML

Then train with:
    python scripts/train.py --dataset data/merged.yaml --base-weights weights/best.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.data.merge import (
    generate_merged_yaml,
    get_merged_class_distribution,
    load_class_mappings,
    merge_dataset,
)
from src.utils.logging_setup import setup_logging
from src.config import load_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge aerial wildlife datasets into a unified training set"
    )
    parser.add_argument("--waid", type=str, default=None, help="Path to WAID/WAID directory")
    parser.add_argument("--aed", type=str, default=None, help="Path to AED dataset root")
    parser.add_argument("--liege", type=str, default=None, help="Path to Liege dataset root")
    parser.add_argument("--wildlifemapper", type=str, default=None, help="Path to WildlifeMapper dataset root (lion, elephant, 20 species)")
    parser.add_argument("--mmla", type=str, default=None, help="Path to MMLA dataset root")
    parser.add_argument(
        "--output", type=str, default="data/merged",
        help="Output directory for merged dataset (default: data/merged)",
    )
    parser.add_argument(
        "--mmla-sample", type=int, default=10,
        help="Keep every Nth frame from MMLA video data (default: 10)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config override file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg)

    if not any([args.waid, args.aed, args.liege, args.wildlifemapper, args.mmla]):
        print("No datasets specified. Run with at least --waid.")
        print("For download instructions: python scripts/prepare_datasets.py")
        sys.exit(1)

    # Load class mappings from config
    mapping_path = PROJECT_ROOT / "config" / "merged_classes.yaml"
    mappings = load_class_mappings(mapping_path)

    with open(mapping_path, encoding="utf-8") as f:
        merged_cfg = yaml.safe_load(f)
    unified_classes: list[str] = merged_cfg["unified_classes"]

    output_dir = PROJECT_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    total_images = 0

    # ── WAID ────────────────────────────────────────────────────────────────
    if args.waid:
        waid_root = Path(args.waid)
        if not waid_root.exists():
            logger.error("WAID path not found: %s", waid_root)
            sys.exit(1)
        logger.info("Merging WAID from %s ...", waid_root)
        stats = merge_dataset(
            dataset_name="waid",
            dataset_root=waid_root,
            mapping=mappings.get("waid", {}),
            output_dir=output_dir,
            split_map={"val": "valid"},  # WAID uses 'valid' not 'val'
        )
        total_images += stats["total_images"]
        logger.info("WAID: %d images merged", stats["total_images"])

    # ── AED ─────────────────────────────────────────────────────────────────
    if args.aed:
        aed_root = Path(args.aed)
        if not aed_root.exists():
            logger.error("AED path not found: %s", aed_root)
            sys.exit(1)
        logger.info("Merging AED from %s ...", aed_root)
        stats = merge_dataset(
            dataset_name="aed",
            dataset_root=aed_root,
            mapping=mappings.get("aed", {}),
            output_dir=output_dir,
        )
        total_images += stats["total_images"]
        logger.info("AED: %d images merged", stats["total_images"])

    # ── Liege ────────────────────────────────────────────────────────────────
    if args.liege:
        liege_root = Path(args.liege)
        if not liege_root.exists():
            logger.error("Liege path not found: %s", liege_root)
            sys.exit(1)
        logger.info("Merging Liege from %s ...", liege_root)
        stats = merge_dataset(
            dataset_name="liege",
            dataset_root=liege_root,
            mapping=mappings.get("liege", {}),
            output_dir=output_dir,
        )
        total_images += stats["total_images"]
        logger.info("Liege: %d images merged", stats["total_images"])

    # ── WildlifeMapper ────────────────────────────────────────────────────────
    if args.wildlifemapper:
        wm_root = Path(args.wildlifemapper)
        if not wm_root.exists():
            logger.error("WildlifeMapper path not found: %s", wm_root)
            sys.exit(1)
        wm_mapping = mappings.get("wildlifemapper", {})
        if not wm_mapping:
            logger.warning(
                "WildlifeMapper class mapping is incomplete in config/merged_classes.yaml. "
                "Check classes.txt in the dataset and update all 20 class IDs."
            )
        logger.info("Merging WildlifeMapper from %s ...", wm_root)
        stats = merge_dataset(
            dataset_name="wildlifemapper",
            dataset_root=wm_root,
            mapping=wm_mapping,
            output_dir=output_dir,
        )
        total_images += stats["total_images"]
        logger.info("WildlifeMapper: %d images merged", stats["total_images"])

    # ── MMLA ─────────────────────────────────────────────────────────────────
    if args.mmla:
        mmla_root = Path(args.mmla)
        if not mmla_root.exists():
            logger.error("MMLA path not found: %s", mmla_root)
            sys.exit(1)
        mmla_mapping = mappings.get("mmla", {})
        if not mmla_mapping:
            logger.warning(
                "MMLA class mapping is empty in config/merged_classes.yaml. "
                "Check classes.txt in the MMLA dataset and update the mapping first."
            )
        logger.info(
            "Merging MMLA from %s (every %dth frame) ...",
            mmla_root, args.mmla_sample,
        )
        stats = merge_dataset(
            dataset_name="mmla",
            dataset_root=mmla_root,
            mapping=mmla_mapping,
            output_dir=output_dir,
            frame_sample=args.mmla_sample,
        )
        total_images += stats["total_images"]
        logger.info("MMLA: %d images merged", stats["total_images"])

    # ── Generate YAML ────────────────────────────────────────────────────────
    yaml_path = generate_merged_yaml(
        output_dir=output_dir,
        unified_classes=unified_classes,
        yaml_path=PROJECT_ROOT / "data" / "merged.yaml",
    )

    # ── Class distribution ───────────────────────────────────────────────────
    dist = get_merged_class_distribution(output_dir, unified_classes)

    print("\n" + "=" * 60)
    print("  Merged Dataset Summary")
    print("=" * 60)
    print(f"  Total images : {total_images:,}")
    print(f"  Output dir   : {output_dir}")
    print(f"  YAML         : {yaml_path}")
    print("\n  Class distribution (train split):")
    for cls, count in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"    {cls:<12s} {count:>8,}")
    print("=" * 60)
    print("\nTo train on merged dataset:")
    print("  python scripts/train.py --dataset data/merged.yaml --base-weights weights/best.pt")


if __name__ == "__main__":
    main()
