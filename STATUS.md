# Malilangwe Wildlife Detector — Project Status

## What This Project Is
YOLOv11-based aerial wildlife detection pipeline for the Malilangwe Trust, Zimbabwe. Detects and classifies animals from drone footage. Built to eventually replace manual wildlife population surveys.

## Current State (as of April 2026)

### Training
- Fine-tuning YOLOv11n on the WAID dataset (14,366 aerial wildlife images, 6 classes)
- Completed: ~90 epochs across 3 Colab sessions (30 epochs each)
- Next session: 30 more epochs → 120 total (set `RESUME=True` in notebook)
- Weights location: `weights/best.pt` and `weights/last.pt` locally, also in Google Drive under `My Drive/malilangwe_weights/`
- Early stopping patience: 15 epochs — training will auto-stop if no improvement

### Detection Results So Far
- 30 epochs: detected most elephants in test image, missed ~4
- 60 epochs: missed ~3
- Labels show "sheep/cattle" because WAID doesn't have elephant class — detection positions are correct, labels will be fixed when Malilangwe-specific data is trained
- Tested locally with `python scripts/detect.py --source path/to/image.jpg --show`

### Codebase (all on branch `claude/code-review-assessment-WDnuF`)
| File | What it does |
|---|---|
| `scripts/detect.py` | CLI detection — run this to test images |
| `scripts/train.py` | Fine-tuning script (used via Colab notebook) |
| `scripts/evaluate.py` | Evaluation — run after training to get mAP scores |
| `notebooks/train_waid_colab.ipynb` | Colab notebook for GPU training |
| `src/detection/detector.py` | YOLOv11 wrapper class |
| `src/tracking/tracker.py` | BoT-SORT tracking interface (built, not yet tested on video) |
| `src/data/dataset.py` | WAID dataset validation and YAML generation |
| `src/utils/visualization.py` | Bounding box drawing |
| `config/default.yaml` | All settings — confidence, epochs, augmentation, paths |
| `data/waid.yaml` | Ultralytics dataset config |

## Roadmap

### Done
- [x] Project structure and config system
- [x] Detection module (YOLOv11 wrapper)
- [x] CLI scripts: detect, train, evaluate
- [x] Dataset validation and tooling
- [x] Visualization utilities
- [x] BoT-SORT tracking interface
- [x] Colab training notebook with Drive backup and resume support
- [x] WAID fine-tuning in progress (~90 epochs done)

### Next
- [ ] Complete WAID training to 120 epochs
- [ ] Run `python scripts/evaluate.py --weights weights/best.pt` to get official mAP score
- [ ] Build `scripts/track.py` — wire tracker to a CLI for video input
- [ ] Collect Malilangwe-specific drone footage (elephant, buffalo, lion, leopard, etc.)
- [ ] Label Malilangwe footage and retrain
- [ ] Edge deployment (ONNX export → Jetson Nano / Raspberry Pi)

## How to Run Locally
```bash
# Install dependencies (once)
pip install -r requirements.txt

# Test detection on an image
python scripts/detect.py --source "path/to/image.jpg" --show

# Evaluate model
python scripts/evaluate.py --weights weights/best.pt

# Train (use Colab notebook instead — no local GPU)
# notebooks/train_waid_colab.ipynb
```

## Colab Training Workflow
1. Open `notebooks/train_waid_colab.ipynb` from branch `claude/code-review-assessment-WDnuF`
2. Set `Runtime → Change runtime type → T4 GPU`
3. Settings cell: `EPOCHS=30`, `BATCH=8`, `RESUME=True/False`
4. Run all — weights auto-save to Google Drive (`My Drive/malilangwe_weights/`)
5. Next session: set `RESUME=True`, loads `waid_last.pt` from Drive automatically

## WAID Dataset Classes
| ID | Class | Train instances |
|----|-------|----------------|
| 0 | Sheep | 91,496 |
| 1 | Cattle | 44,245 |
| 2 | Seal | 15,762 |
| 3 | Camelus | 4,676 |
| 4 | Kiang | 3,312 |
| 5 | Zebra | 3,792 |

Note: No Malilangwe-specific species yet. Retraining on local data is the key next milestone.
