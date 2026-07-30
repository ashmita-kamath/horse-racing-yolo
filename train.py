from ultralytics import YOLO

model = YOLO("yolo11m.pt")
model.train(
    data="ICT320-Y12-HorseRacing-dataset-1/data.yaml",
    epochs=200, imgsz=960, batch=4,
    device="mps", patience=50, amp=False,
    project="horse_racing", name="full_m_960",
)
model.val()
