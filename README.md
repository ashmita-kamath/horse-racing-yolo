# Horse Racing Detection - YOLOv11

Object detection model to detect horses in horse racing footage, built on the ICT320-Y12-HorseRacing dataset using YOLOv11.

## Dataset
- Source: ICT320-Y12-HorseRacing (Roboflow), single class: racing-horse
- 641 images total - 452 train / 130 valid / 59 test
- A sample (~15 images + labels per split) and data.yaml are included under sample_data/ to show the structure.

## Models Trained
YOLOv11n - image size 640, 100 epochs (patience 20): mAP50 0.931, Precision 0.924, Recall 0.875, mAP50-95 0.495
YOLOv11m - image size 960, 200 epochs (early stop at 77): mAP50 0.946, Precision 0.923, Recall 0.897, mAP50-95 0.503

The YOLOv11m run was trained on Google Colab (T4 GPU); the notebook is included as yolov11n.ipynb.

## Files
- train.py - training script
- test_images.py - runs the model on the test images and saves annotated outputs
- video.py / paddock_test.py - runs the model on a video recording
- see.py - quick inference on a folder of images
- yolov11n.ipynb - Colab notebook used for the YOLOv11m training run
- sample_data/ - sample train/valid/test images + labels and data.yaml

## How to run
Install: pip install ultralytics roboflow
Train: python train.py
Test: python test_images.py

## Notes on paddock footage
The model performs well on racing footage (~0.6-0.9 confidence) but struggles on paddock footage where horses stand next to handlers (confidence drops to ~0.44-0.49, boxes include the handler). This is expected since the training data is all racing shots. Improving paddock detection requires paddock training data (annotation / augmentation), the planned next step.

## Data / logging
Paddock video (too large for GitHub): https://drive.google.com/file/d/19NOzkWYh-q8LNhnC_vVd1I2XDt0LVwNV/view?usp=sharing