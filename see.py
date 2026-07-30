from ultralytics import YOLO
model = YOLO("/Users/ashmitakamath/runs/detect/horse_racing/full/weights/best.pt")
results = model.predict(
    source="ICT320-Y12-HorseRacing-dataset-1/test/images",
    conf=0.4, save=True,
)
print("Saved to:", results[0].save_dir)
