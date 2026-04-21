"""
Multi-dataset merge utilities for Phase A+ training.

Remaps class IDs from multiple aerial wildlife datasets to the unified
Prometheus class schema defined in config/merged_classes.yaml, then
combines images and labels into a single dataset directory.

Usage (via scripts/merge_datasets.py):
    python scripts/merge_datasets.py --waid WAID/WAID --aed path/to/AED
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_SPLITS = ("train", "val", "test")
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_class_mappings(config_path: str | Path) -> dict[str, dict[int, int]]:
    """Load per-dataset class remapping from merged_classes.yaml.

    Returns:
        Dict of dataset_name → {src_class_id: unified_class_id}.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Merged class config not found: {path}")

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    mappings: dict[str, dict[int, int]] = {}
    for dataset, raw in cfg.get("dataset_mappings", {}).items():
        mappings[dataset] = {int(k): int(v) for k, v in (raw or {}).items()}
    return mappings


def remap_label_file(
    src: Path,
    dst: Path,
    mapping: dict[int, int],
    dataset_name: str,
) -> tuple[int, int]:
    """Read a YOLO label file, remap class IDs, write to dst.

    Lines with unmapped class IDs are skipped with a warning.

    Returns:
        (kept, skipped) line counts.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    kept = skipped = 0

    with open(src, encoding="utf-8") as f:
        lines = f.readlines()

    out_lines: list[str] = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        src_id = int(parts[0])
        if src_id not in mapping:
            logger.warning(
                "[%s] Unmapped class ID %d in %s — skipping line",
                dataset_name, src_id, src.name,
            )
            skipped += 1
            continue
        parts[0] = str(mapping[src_id])
        out_lines.append(" ".join(parts) + "\n")
        kept += 1

    with open(dst, "w", encoding="utf-8") as f:
        f.writelines(out_lines)

    return kept, skipped


def _find_image(label_path: Path, images_dir: Path) -> Path | None:
    """Find the image file corresponding to a label file."""
    for ext in _IMG_EXTS:
        candidate = images_dir / (label_path.stem + ext)
        if candidate.exists():
            return candidate
    return None


def merge_dataset(
    dataset_name: str,
    dataset_root: Path,
    mapping: dict[int, int],
    output_dir: Path,
    split_map: dict[str, str] | None = None,
    frame_sample: int = 1,
) -> dict[str, Any]:
    """Merge one dataset into the unified output directory.

    Args:
        dataset_name:  Short name used as filename prefix (e.g. "waid").
        dataset_root:  Root of the source dataset (must contain images/ and labels/).
        mapping:       Class ID remapping {src_id: unified_id}.
        output_dir:    Destination root (data/merged/).
        split_map:     Override split directory names, e.g. {"val": "valid"}.
        frame_sample:  Keep every Nth image (use 10 for MMLA video frames).

    Returns:
        Stats dict with per-split image/label counts and skipped lines.
    """
    split_map = split_map or {}
    stats: dict[str, Any] = {"splits": {}, "total_images": 0, "skipped_lines": 0}

    for split in _SPLITS:
        src_split = split_map.get(split, split)
        src_img_dir = dataset_root / "images" / src_split
        src_lbl_dir = dataset_root / "labels" / src_split

        if not src_img_dir.exists() or not src_lbl_dir.exists():
            logger.warning("[%s] Split '%s' not found, skipping", dataset_name, src_split)
            continue

        dst_img_dir = output_dir / "images" / split
        dst_lbl_dir = output_dir / "labels" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        label_files = sorted(p for p in src_lbl_dir.iterdir() if p.suffix == ".txt")
        copied = skipped_lines = 0

        for i, lbl_file in enumerate(label_files):
            # Frame subsampling for video datasets
            if i % frame_sample != 0:
                continue

            img_file = _find_image(lbl_file, src_img_dir)
            if img_file is None:
                logger.warning("[%s] No image for label %s", dataset_name, lbl_file.name)
                continue

            # Prefix filename with dataset name to avoid collisions
            prefix = f"{dataset_name}_{lbl_file.stem}"
            dst_lbl = dst_lbl_dir / f"{prefix}.txt"
            dst_img = dst_img_dir / f"{prefix}{img_file.suffix}"

            kept, sk = remap_label_file(lbl_file, dst_lbl, mapping, dataset_name)
            skipped_lines += sk

            if kept > 0:
                shutil.copy2(img_file, dst_img)
                copied += 1
            else:
                dst_lbl.unlink(missing_ok=True)

        stats["splits"][split] = {"images": copied}
        stats["total_images"] += copied
        stats["skipped_lines"] += skipped_lines
        logger.info(
            "[%s] %s split: %d images merged, %d label lines skipped",
            dataset_name, split, copied, skipped_lines,
        )

    return stats


def generate_merged_yaml(
    output_dir: Path,
    unified_classes: list[str],
    yaml_path: str | Path = "data/merged.yaml",
) -> Path:
    """Write the Ultralytics dataset YAML for the merged dataset."""
    out = Path(yaml_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    ds: dict[str, Any] = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(unified_classes),
        "names": unified_classes,
    }
    with open(out, "w", encoding="utf-8") as f:
        yaml.dump(ds, f, default_flow_style=False, sort_keys=False)

    logger.info("Merged dataset YAML written to %s", out.resolve())
    return out.resolve()


def get_merged_class_distribution(output_dir: Path, class_names: list[str]) -> dict[str, int]:
    """Count per-class instances across the merged training split."""
    lbl_dir = output_dir / "labels" / "train"
    counts = {name: 0 for name in class_names}

    if not lbl_dir.exists():
        return counts

    for lbl_file in lbl_dir.iterdir():
        if lbl_file.suffix != ".txt":
            continue
        with open(lbl_file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                if 0 <= cls_id < len(class_names):
                    counts[class_names[cls_id]] += 1

    return counts
