from roboflow import Roboflow
from ultralytics import YOLO

rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("ashmita-kamath131207-gmail-com").project("ict320-y12-horseracing-dataset-b7ipi-ofr6d")
dataset = project.version(1).download("yolov11")

model = YOLO("yolo11m.pt")
model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=200, imgsz=960, batch=8,
    device="mps", patience=50,
    project="horse_racing", name="full_m_960",
)
model.val()
