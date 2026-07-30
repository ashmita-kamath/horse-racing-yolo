from ultralytics import YOLO
model = YOLO("best_m_960.pt")
results = model.predict(
    source="paddock.mov",
    conf=0.7, iou=0.45, imgsz=960, save=True, stream=True,
)
for r in results:
    pass
print("Done")
