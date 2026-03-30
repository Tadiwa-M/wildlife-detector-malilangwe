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
Wildlife Detector/
├── config/
│   ├── default.yaml          # Main configuration
│   └── waid.yaml             # Dataset config for Ultralytics
├── src/
│   ├── detection/
│   │   └── detector.py       # WildlifeDetector class
│   ├── tracking/             # BoT-SORT integration (planned)
│   ├── data/                 # Dataset loading & preprocessing
│   └── utils/                # Visualization, logging, helpers
├── scripts/
│   └── detect.py             # CLI detection runner
├── tests/                    # Unit tests
├── WAID/                     # Dataset (not tracked in git)
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/Tadiwa-M/wildlife-detector-malilangwe.git
cd wildlife-detector-malilangwe
pip install -r requirements.txt
```

### Run Detection

```bash
# On a single image
python scripts/detect.py path/to/image.jpg

# On a folder of images
python scripts/detect.py path/to/folder/

# With custom confidence threshold
python scripts/detect.py path/to/image.jpg --conf 0.15

# With specific weights
python scripts/detect.py path/to/image.jpg --weights runs/train/best.pt
```

### Train on WAID Dataset

```bash
# Download the dataset
git clone https://github.com/xiaohuicui/WAID.git

# Train (GPU recommended — use Google Colab)
python scripts/train.py --config config/default.yaml
```

## Dataset

**WAID (Wildlife Aerial Images from Drone)** — 14,375 UAV aerial images across 6 species:

| Class | Species |
|-------|---------|
| 0 | Sheep |
| 1 | Cattle |
| 2 | Seal |
| 3 | Camelus |
| 4 | Kiang |
| 5 | Zebra |

Source: [WAID GitHub](https://github.com/xiaohuicui/WAID) · [Paper](https://www.mdpi.com/2076-3417/13/18/10397)

## Architecture

- **Detection:** YOLOv11 (Ultralytics) — real-time object detection optimised for small aerial targets
- **Tracking:** BoT-SORT — multi-object tracking for video sequences (planned)
- **Config:** YAML-driven — no hardcoded paths or hyperparameters
- **Design:** Modular, single-responsibility modules, type-hinted, production-aware

## Roadmap

- [x] Project structure & config system
- [x] Detection module with config-driven inference
- [ ] Fine-tune YOLOv11 on WAID dataset
- [ ] Evaluation & metrics module
- [ ] BoT-SORT tracking integration for video
- [ ] Edge deployment testing (Raspberry Pi / Jetson Nano)
- [ ] Malilangwe-specific species classes

## About Malilangwe

The [Malilangwe Trust](https://www.malilangwe.org/) manages the Malilangwe Wildlife Reserve in southeastern Zimbabwe — over 130,000 acres of protected savanna, woodland, and wetland. This project aims to support their ecological monitoring efforts with scalable, automated aerial survey tools.

## References

- [WAID: A Large-Scale Dataset for Wildlife Detection with Drones](https://www.mdpi.com/2076-3417/13/18/10397) (Cui et al., 2023)
- [YOLOv11-Lite: Wildlife Detection from Drone Images](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1777913/full) (2026)
- [Ultralytics YOLOv11 Documentation](https://docs.ultralytics.com/)

## License

MIT
