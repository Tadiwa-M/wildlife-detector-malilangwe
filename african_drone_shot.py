from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.predict(
    source="waid_samples/",
    save=True,
    conf=0.15
)

for r in results:
    print(f"{r.path}: {len(r.boxes)} detections")

print("\nDone! Check runs/detect/ folder.")