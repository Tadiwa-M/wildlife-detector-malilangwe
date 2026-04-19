# Prometheus — Project Status

## What This Is
**Prometheus** is an aerial intelligence platform for African land stewards, starting with wildlife detection for the Malilangwe Trust in south-eastern Zimbabwe. The platform processes drone footage using YOLOv11 + BoT-SORT and surfaces results in a web dashboard with species counts, population charts, anomaly flags, and map view.

One-line pitch: *"Prometheus turns aerial imagery into actionable intelligence for African land stewards."*

---

## Current Phase: A — WAID Fine-Tuning

### Training Progress
| Session | Epochs | Status |
|---|---|---|
| 1 | 1–30 | Done |
| 2 | 31–60 | Done |
| 3 | 61–90 | Done |
| 4 | 91–120 | In progress |

- Weights: `weights/best.pt` and `weights/last.pt` locally + `My Drive/malilangwe_weights/`
- Early stopping: patience=15 — auto-stops if no improvement
- Target: >90% mAP@0.5 on WAID test set

### Detection Results
- 30 epochs: ~4 missed elephants on test image
- 60 epochs: ~3 missed
- Labels show "sheep/cattle" — WAID doesn't have elephant class. Detection positions are correct. Labels fixed in Phase B.

---

## Codebase (branch: `claude/code-review-assessment-WDnuF`)

| File | What it does |
|---|---|
| `scripts/detect.py` | CLI detection — test images locally |
| `scripts/train.py` | Training script (run via Colab notebook) |
| `scripts/evaluate.py` | Evaluation — mAP, precision, recall |
| `notebooks/train_waid_colab.ipynb` | Colab GPU training with Drive backup + resume |
| `src/detection/detector.py` | YOLOv11 wrapper |
| `src/tracking/tracker.py` | BoT-SORT interface (built, not yet tested on video) |
| `src/data/dataset.py` | WAID validation and YAML generation |
| `src/utils/visualization.py` | Bounding box drawing |
| `config/default.yaml` | All settings |
| `data/waid.yaml` | Ultralytics dataset config |
| `STATUS.md` | This file — update after each milestone |
| `BACKLOG.md` | Ideas and future scope not in current phase |

---

## Roadmap

### Phase A — WAID Training (now)
- [x] Project structure, config, detection module
- [x] CLI scripts: detect, train, evaluate
- [x] Colab training notebook with Drive backup
- [ ] Complete 120 epochs
- [ ] Run `python scripts/evaluate.py --weights weights/best.pt` — get official mAP
- [ ] Write results to RESULTS.md

### Phase A+ — Multi-Dataset Training (stretch, after Phase A)
Merge WAID with additional aerial wildlife datasets for better generalization:
- **AED** (Aerial Elephant Dataset): 15,511 elephants, 2,101 images
- **MMLA**: 811K annotations, 6 species (subsample every 10th frame)
- **Liege African Mammals**: Buffalo, elephant, kob, topi, warthog, waterbuck

Unified class schema: elephant, zebra, giraffe, buffalo, antelope, other. Needs `src/merge_datasets.py`.

### Phase 2 — FastAPI Backend
- Async video processing with Celery + Redis
- Postgres for storing detection results
- Endpoints: upload video, poll job status, get results
- Runs the detector/tracker pipeline as background jobs

### Phase 3 — React Dashboard (the demo milestone)
- Upload drone footage via browser
- Real-time processing status
- Results: annotated video, species counts, population charts, map view (leaflet or mapbox)
- Stack: React + Tailwind + shadcn/ui
- **This is what Brightlands judges and Erik (WUR) see**
- Target: 90-second demo video showing end-to-end flow

### Phase B — Malilangwe-Specific Transfer Learning (future)
- Requires labelled aerial imagery from Bruce at Malilangwe Trust (email sent, no response yet)
- 19 priority species: elephant, black/white rhino, lion, leopard, buffalo, giraffe, impala, kudu, sable, roan, hartebeest, nyala, waterbuck, wildebeest, wild dog, cheetah, hippo
- Base model: `best.pt` from Phase A (not yolo11n.pt)
- 30 epochs, lower LR, freeze 10 backbone layers

### Phase 5+ — Edge Deployment
- ONNX/TensorRT export, INT8 quantisation
- Jetson Nano/Orin on drone or ground station
- Not in MVP

---

## How to Run Locally
```bash
pip install -r requirements.txt

# Test detection
python scripts/detect.py --source "path/to/image.jpg" --show

# Evaluate model
python scripts/evaluate.py --weights weights/best.pt
```

## Colab Training Workflow
1. Open `notebooks/train_waid_colab.ipynb` — branch `claude/code-review-assessment-WDnuF`
2. `Runtime → Change runtime type → T4 GPU`
3. Settings: `EPOCHS=30`, `BATCH=8`, `RESUME=True` (after first session)
4. Run all — weights auto-save to `My Drive/malilangwe_weights/`

---

## Key Technical Decisions
- **PyTorch 2.6.0+cpu locally** — 2.11.0 crashes c10.dll on Anaconda/Windows
- **WAID classes ≠ Malilangwe species** — generic labels, fixed in Phase B
- **Config override pattern** — `config/local.yaml` (git-ignored) for machine-specific paths
- **Model is not the moat** — defensibility comes from multi-dataset generalization, per-reserve fine-tunes, field deployment, and conservancy relationships

---

## Project Priorities
1. Portfolio value — make Swae hireable at ag-robotics/CV companies (Lely, WUR/Erik Pekkeriet)
2. Brightlands Startup Challenge — forcing function for the demo
3. Startup viability — preserved as longer-term possibility
4. Scope discipline — if it's not Phase 1–3, it goes in BACKLOG.md
