"""
Data-loading utilities for the WAID dataset.

The WAID dataset ships pre-split into train/valid/test directories under
both images/ and labels/. This module validates that structure and generates
the ``dataset.yaml`` file required by Ultralytics for training.

Usage:
    from src.config import load_config
    from src.data.dataset import validate_dataset, generate_dataset_yaml

    cfg = load_config()
    stats = validate_dataset(cfg)
    yaml_path = generate_dataset_yaml(cfg)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from src.config import Config

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def validate_dataset(cfg: Config) -> dict[str, Any]:
    """Verify WAID dataset integrity and report per-split statistics.

    Checks that every split directory exists, counts images and labels,
    and flags orphans (images without labels or vice-versa).

    Args:
        cfg: Pipeline configuration.

    Returns:
        Dict with per-split counts and overall totals.

    Raises:
        FileNotFoundError: If the dataset root or expected directories are missing.
    """
    root = Path(str(cfg.paths.dataset_root))
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {root}\n"
            "Clone the WAID repo or update paths.dataset_root in config."
        )

    split_dirs = cfg.dataset.raw.get("split_dirs", {})
    stats: dict[str, Any] = {"splits": {}, "total_images": 0, "total_labels": 0}

    for split_name, dir_name in split_dirs.items():
        img_dir = root / "images" / dir_name
        lbl_dir = root / "labels" / dir_name

        if not img_dir.exists():
            logger.warning("Missing image directory: %s", img_dir)
            continue
        if not lbl_dir.exists():
            logger.warning("Missing label directory: %s", lbl_dir)
            continue

        img_stems = {
            p.stem for p in img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        }
        lbl_stems = {
            p.stem for p in lbl_dir.iterdir()
            if p.is_file() and p.suffix == ".txt"
        }

        split_stats = {
            "images": len(img_stems),
            "labels": len(lbl_stems),
            "images_without_labels": len(img_stems - lbl_stems),
            "labels_without_images": len(lbl_stems - img_stems),
        }
        stats["splits"][split_name] = split_stats
        stats["total_images"] += split_stats["images"]
        stats["total_labels"] += split_stats["labels"]

        logger.info(
            "  %-6s — images: %4d  labels: %4d  orphan_img: %d  orphan_lbl: %d",
            split_name,
            split_stats["images"],
            split_stats["labels"],
            split_stats["images_without_labels"],
            split_stats["labels_without_images"],
        )

    logger.info(
        "Dataset totals — images: %d, labels: %d",
        stats["total_images"],
        stats["total_labels"],
    )
    return stats


def generate_dataset_yaml(
    cfg: Config,
    output_path: str | Path = "data/waid.yaml",
) -> Path:
    """Generate the ``dataset.yaml`` file required by Ultralytics for training.

    Points directly at the pre-split WAID directories.

    Args:
        cfg:          Pipeline configuration.
        output_path:  Where to write the YAML file.

    Returns:
        Absolute path to the generated YAML file.
    """
    root = Path(str(cfg.paths.dataset_root)).resolve()
    split_dirs = cfg.dataset.raw.get("split_dirs", {})

    ds_yaml: dict[str, Any] = {
        "path": str(root),
        "train": f"images/{split_dirs.get('train', 'train')}",
        "val": f"images/{split_dirs.get('val', 'valid')}",
        "test": f"images/{split_dirs.get('test', 'test')}",
        "nc": int(cfg.dataset.num_classes),
        "names": list(cfg.dataset.class_names),
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        yaml.dump(ds_yaml, fh, default_flow_style=False, sort_keys=False)

    logger.info("Dataset YAML written to %s", out.resolve())
    return out.resolve()


def get_class_distribution(cfg: Config, split: str = "train") -> dict[str, int]:
    """Count per-class instances across all label files in a split.

    Reads every .txt label file and tallies class IDs.

    Args:
        cfg:   Pipeline configuration.
        split: Which split to scan (``"train"``, ``"val"``, ``"test"``).

    Returns:
        Dict mapping class name → instance count.
    """
    root = Path(str(cfg.paths.dataset_root))
    split_dirs = cfg.dataset.raw.get("split_dirs", {})
    dir_name = split_dirs.get(split, split)
    lbl_dir = root / "labels" / dir_name

    class_names = list(cfg.dataset.class_names)
    counts: dict[str, int] = {name: 0 for name in class_names}

    if not lbl_dir.exists():
        logger.warning("Label directory not found: %s", lbl_dir)
        return counts

    for lbl_file in lbl_dir.iterdir():
        if not lbl_file.suffix == ".txt":
            continue
        with open(lbl_file, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                if 0 <= cls_id < len(class_names):
                    counts[class_names[cls_id]] += 1

    logger.info("Class distribution (%s): %s", split, counts)
    return counts
