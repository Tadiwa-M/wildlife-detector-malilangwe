"""
Dataset preparation guide for Phase A+ multi-dataset training.

Run this script to see exactly what to download and how to structure each
dataset before running merge_datasets.py.

Usage:
    python scripts/prepare_datasets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════════════╗
║         Prometheus — Phase A+ Dataset Preparation Guide             ║
╚══════════════════════════════════════════════════════════════════════╝

Before running merge_datasets.py, download and structure each dataset
as described below. All datasets use YOLO format (class_id cx cy w h).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 1. WAID — Wildlife Aerial Images from Drone
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Already included in this repo (labels only). Images:
  git clone https://github.com/xiaohuicui/WAID.git

Expected structure:
  WAID/WAID/
    images/
      train/   valid/   test/
    labels/
      train/   valid/   test/

Pass to merge script as: --waid WAID/WAID

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 2. AED — Aerial Elephant Dataset
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
15,511 elephant annotations across 2,101 aerial images.
Single class: elephant (ID 0).

Download:
  https://zenodo.org/record/3234780
  → Download aed.zip, extract it.

Convert to YOLO format if needed (AED ships as Pascal VOC XML):
  pip install pylabel
  python -c "
from pylabel import importer
ds = importer.ImportVOC('path/to/AED/annotations')
ds.export.ExportToYoloV5(output_path='path/to/AED/yolo')
"

Expected structure after conversion:
  AED/
    images/
      train/   val/   test/
    labels/
      train/   val/   test/

If AED has no pre-made splits, use an 80/10/10 split:
  python -c "
import os, shutil, random
from pathlib import Path

src = Path('path/to/AED/yolo')
imgs = list((src / 'images').glob('*.jpg'))
random.seed(42)
random.shuffle(imgs)
n = len(imgs)
splits = {'train': imgs[:int(n*0.8)], 'val': imgs[int(n*0.8):int(n*0.9)], 'test': imgs[int(n*0.9):]}
for split, files in splits.items():
    (src / 'images' / split).mkdir(parents=True, exist_ok=True)
    (src / 'labels' / split).mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.move(str(f), src / 'images' / split / f.name)
        lbl = src / 'labels' / (f.stem + '.txt')
        if lbl.exists():
            shutil.move(str(lbl), src / 'labels' / split / lbl.name)
"

Pass to merge script as: --aed path/to/AED

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 3. Liege African Mammals
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Aerial imagery of 7 African species: zebra, elephant, buffalo,
kob, topi, warthog, waterbuck.

Download from LILA.science:
  https://lila.science/datasets/wcscameratraps
  (search: "Liege" or "African mammals aerial")

Alternatively: https://github.com/aerial-wildlife/liege-african-mammals

Expected structure:
  liege/
    images/
      train/   val/   test/
    labels/
      train/   val/   test/
    classes.txt   ← verify class order matches config/merged_classes.yaml

IMPORTANT: Check classes.txt and verify the class ID order matches
what's in config/merged_classes.yaml under dataset_mappings.liege.
Update the mapping if the order differs.

Pass to merge script as: --liege path/to/liege

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 4. WildlifeMapper — SKIP FOR V1 (access required)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The full dataset requires institutional access approval which is not yet
granted. The 4TU download link is dead. Do NOT pass --wildlifemapper
when running merge_datasets.py for v1 training.

GitHub: https://github.com/UCSB-VRL/WildlifeMapper
  (request access here if you want to pursue lion/giraffe for v2)

When access is granted:
  1. Download and structure as images/train, val, test + labels/
  2. Check classes.txt and verify IDs in config/merged_classes.yaml
  3. Re-add "giraffe" and "lion" to unified_classes in merged_classes.yaml
  4. Update wildlifemapper mapping IDs 2 and 3 (currently mapped to other)
  5. Re-run merge and retrain

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 5. MMLA — SKIP FOR V1 (not yet publicly available)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The LILA.science download link is dead — dataset not yet published.
Do NOT pass --mmla when running merge_datasets.py for v1 training.

Paper: https://arxiv.org/pdf/2504.07744
  (check here for a data release announcement)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Once all datasets are ready, run:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # v1 — WAID + AED + Liege (all freely available):
  python scripts/merge_datasets.py \\
      --waid WAID/WAID \\
      --aed path/to/AED \\
      --liege path/to/liege

  # Minimum (WAID + AED only, if Liege not ready):
  python scripts/merge_datasets.py --waid WAID/WAID --aed path/to/AED

  # Skip --wildlifemapper and --mmla for now — both datasets are unavailable.

Output will be in data/merged/ with a ready-to-use data/merged.yaml.
"""


def main() -> None:
    print(INSTRUCTIONS)


if __name__ == "__main__":
    main()
