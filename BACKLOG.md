# Backlog

Ideas and future scope that are NOT in the current MVP (Phases 1–3). Do not implement until the demo is done.

---

## Model & Training
- Phase A+ multi-dataset merge (AED + MMLA + Liege) — needs `src/merge_datasets.py` with unified class schema
- Phase B Malilangwe-specific fine-tune — blocked on data from Bruce
- Focal loss tuning for class imbalance (sheep 28x kiang)
- Oversampling rare classes in WAID
- YOLOv11s or YOLOv11m upgrade after validating pipeline on nano

## Platform Expansion (pitch narrative only)
- Crop health monitoring (Sentinel-2, NDVI)
- Rangeland condition assessment
- Livestock monitoring
- These are vision slide material, not shipped features

## Edge Deployment
- ONNX/TensorRT export
- INT8 quantisation with calibration dataset
- Jetson Nano/Orin field deployment
- Near-real-time inference on drone

## Dashboard Features (post-MVP)
- Historical trend charts across survey dates
- Anomaly detection and alerts (animal in unusual location, population drop)
- Per-species heatmaps
- Multi-reserve support
- Export to PDF report for ecologists

## Infrastructure
- Docker + docker-compose for local dev setup
- CI/CD with GitHub Actions
- Proper secrets management
- Rate limiting on API endpoints
