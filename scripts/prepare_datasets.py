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
 4. Kuzikus/Namibia Dataset — RHINO, giraffe, eland, ostrich, springbok
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Aerial imagery from Kuzikus Wildlife Reserve, Namibia.
Key species for Prometheus: rhino (black + white), giraffe.

Paper + data: https://peerj.com/articles/13779/

Download the dataset from the paper's supplementary materials or:
  https://doi.org/10.7717/peerj.13779

IMPORTANT: Check classes.txt after downloading and verify the class
order matches config/merged_classes.yaml under dataset_mappings.kuzikus.
The script assumes: rhino=0, giraffe=1, eland=2, ostrich=3, springbok=4.
Update if the actual order differs.

Expected structure:
  kuzikus/
    images/
      train/   val/   test/
    labels/
      train/   val/   test/
    classes.txt

Pass to merge script as: --kuzikus path/to/kuzikus

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 5. WildlifeMapper (CVPR 2024) — LION, elephant, giraffe, zebra + 16 more
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
20-species aerial dataset including lion, Masai giraffe, elephant, zebra.
Key species for Prometheus: lion (only dataset with lion labels).

Download:
  https://data.4tu.nl/articles/dataset/12713903/1

IMPORTANT: 20-class dataset. After downloading, check classes.txt and
update ALL 20 class IDs in config/merged_classes.yaml under
dataset_mappings.wildlifemapper. Currently only the first 6 are mapped —
unmapped classes are skipped with a warning, not silently dropped.

Expected structure:
  wildlifemapper/
    images/
      train/   val/   test/
    labels/
      train/   val/   test/
    classes.txt   ← MUST verify before training

Pass to merge script as: --wildlifemapper path/to/wildlifemapper

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 6. MMLA — Multi-Modal Large Animal Dataset (optional)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
811K annotations from 37 aerial videos, 6 species.
Dense video data — merge script subsamples every 10th frame automatically.

Download:
  https://lila.science/datasets/mmla

After downloading, check classes.txt and update dataset_mappings.mmla
in config/merged_classes.yaml accordingly (currently empty — needs
confirmation of class IDs from the actual dataset).

Expected structure:
  mmla/
    images/
      train/   val/   test/
    labels/
      train/   val/   test/
    classes.txt

Pass to merge script as: --mmla path/to/mmla

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Once all datasets are ready, run:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # Minimum (WAID + AED):
  python scripts/merge_datasets.py --waid WAID/WAID --aed path/to/AED

  # Full merge:
  python scripts/merge_datasets.py \\
      --waid WAID/WAID \\
      --aed path/to/AED \\
      --liege path/to/liege \\
      --mmla path/to/mmla

Output will be in data/merged/ with a ready-to-use data/merged.yaml.
"""


def main() -> None:
    print(INSTRUCTIONS)


if __name__ == "__main__":
    main()
