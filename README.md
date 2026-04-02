# Malilangwe Wildlife Aerial Detection

Automated wildlife detection and tracking from aerial drone footage, built for the **Malilangwe Trust** — a conservation organisation managing one of Zimbabwe's most biodiverse wildlife reserves.

## The Problem

Wildlife population monitoring across large terrain is costly, time-consuming, and error-prone when done manually. Rangers and ecologists need accurate, scalable tools to count animals, track herds, and flag anomalies from aerial surveys.

## The Solution

An ML pipeline that processes drone footage to **detect**, **classify**, and **track** wildlife species in real-time using YOLOv11 object detection and BoT-SORT multi-object tracking.

### Pretrained vs Fine-tuned

A pretrained COCO model sees aerial elephants and calls them "sheep" or "birds" — it knows *something* is there, but can't classify it correctly from a drone perspective. Fine-tuning on aerial wildlife data fixes this:

| | Pretrained (COCO) | Fine-tuned (WAID) |
|---|---|---|
| Aerial elephants | "sheep", "cow", "bird" | "cattle" ✓ |
| Aerial lions | "horse", "sheep" | Coming soon |
| Detection accuracy | ~30% | Target: >85% mAP |

## Project Structure

```
wildlife-detector-malilangwe/
├── config/
│   └── default.yaml          # Master configuration (paths, model, training, tracking)
├── data/
│   └── waid.yaml             # Ultralytics dataset YAML for training
├── src/
│   ├── config.py             # YAML config loader with dot-access & deep merge
│   ├── detection/
│   │   └── detector.py       # WildlifeDetector class (YOLOv11 wrapper)
│   ├── tracking/
│   │   └── tracker.py        # BoT-SORT multi-object tracking
│   ├── data/
│   │   └── dataset.py        # WAID dataset validation & YAML generation
│   └── utils/
│       ├── logging_setup.py  # Structured logging (file + stdout)
│       └── visualization.py  # Bounding box & summary overlay drawing
├── scripts/
│   ├── detect.py             # CLI detection runner
│   ├── train.py              # Fine-tuning on WAID dataset
│   └── evaluate.py           # Model evaluation & metrics
├── tests/
│   ├── test_config.py        # Config system tests
│   └── test_detection.py     # Detection class tests
├── WAID/                     # Dataset labels (images not tracked — too large)
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.10+
- pip
- GPU recommended for training (use Google Colab if needed)

### Installation

```bash
git clone https://github.com/Tadiwa-M/wildlife-detector-malilangwe.git
cd wildlife-detector-malilangwe
pip install -r requirements.txt
```

### Run Detection

```bash
# On a single image
python scripts/detect.py --source path/to/image.jpg

# On a folder of images
python scripts/detect.py --source path/to/folder/

# With custom confidence threshold
python scripts/detect.py --source path/to/image.jpg --conf 0.15

# Display results in a window
python scripts/detect.py --source path/to/image.jpg --show

# Save annotated results
python scripts/detect.py --source path/to/image.jpg --save
```

### Train on WAID Dataset

```bash
# Download the dataset images (labels already included)
git clone https://github.com/xiaohuicui/WAID.git

# Validate dataset integrity
python scripts/train.py --validate-only

# Train (GPU recommended)
python scripts/train.py

# Resume from checkpoint
python scripts/train.py --resume runs/train/weights/last.pt

# Custom config
python scripts/train.py --config config/custom.yaml
```

### Evaluate a Model

```bash
# Evaluate on test split
python scripts/evaluate.py --weights weights/best.pt

# Evaluate on validation split
python scripts/evaluate.py --weights weights/best.pt --split val

# Save evaluation plots
python scripts/evaluate.py --weights weights/best.pt --save-plots
```

## Dataset

**WAID (Wildlife Aerial Images from Drone)** — 14,366 UAV aerial images across 6 species:

| Class | Species | Train instances |
|-------|---------|----------------|
| 0 | Sheep | 91,496 |
| 1 | Cattle | 44,245 |
| 2 | Seal | 15,762 |
| 3 | Camelus | 4,676 |
| 4 | Kiang | 3,312 |
| 5 | Zebra | 3,792 |

> **Note:** There is significant class imbalance (sheep has 28x more instances than kiang). The training config includes class weighting to mitigate this.

Source: [WAID GitHub](https://github.com/xiaohuicui/WAID) · [Paper](https://www.mdpi.com/2076-3417/13/18/10397)

## Architecture

- **Detection:** YOLOv11 (Ultralytics) — real-time object detection optimised for small aerial targets
- **Tracking:** BoT-SORT — multi-object tracking for video sequences
- **Config:** YAML-driven — no hardcoded paths or hyperparameters
- **Design:** Modular, single-responsibility modules, type-hinted

## Roadmap

- [x] Project structure & config system
- [x] Detection module with config-driven inference
- [x] CLI scripts (detect, train, evaluate)
- [x] Dataset validation & statistics tooling
- [x] Visualization utilities
- [x] BoT-SORT tracking interface
- [ ] Fine-tune YOLOv11 on WAID dataset
- [ ] Evaluation & metrics benchmarking
- [ ] Edge deployment testing (Raspberry Pi / Jetson Nano)
- [ ] Malilangwe-specific species classes (elephant, buffalo, lion, etc.)

## About Malilangwe

The [Malilangwe Trust](https://www.malilangwe.org/) manages the Malilangwe Wildlife Reserve in southeastern Zimbabwe — over 130,000 acres of protected savanna, woodland, and wetland. This project aims to support their ecological monitoring efforts with scalable, automated aerial survey tools.

## References

- [WAID: A Large-Scale Dataset for Wildlife Detection with Drones](https://www.mdpi.com/2076-3417/13/18/10397) (Cui et al., 2023)
- [Ultralytics YOLOv11 Documentation](https://docs.ultralytics.com/)

## License

MIT
