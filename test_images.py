from ultralytics import YOLO
model = YOLO("best_m_960.pt")
results = model.predict(
    source="ICT320-Y12-HorseRacing-dataset-1/test/images",
    conf=0.4, iou=0.45, imgsz=960, save=True,
)
print("Saved to:", results[0].save_dir)
