from ultralytics import YOLO
model = YOLO("/Users/ashmitakamath/runs/detect/horse_racing/full/weights/best.pt")
results = model.predict(
    source="paddock.mov",
    conf=0.4, save=True, stream=True,
)
for r in results:
    pass
print("Done")
