# detect.py

from ultralytics import YOLO
import numpy as np
from PIL import Image
import os

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # You can also use yolov8s.pt, yolov8m.pt, etc.

def detect_objects(image_path, target_class="Bottle", save_path=None):
    results = model(image_path)[0]

    detections = []
    for box in results.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        if class_name != target_class:
            continue

        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        detections.append({
            "class": class_name,
            "confidence": round(conf, 3),
            "bbox": [x1, y1, x2, y2]
        })

    # Save the image with boxes if path provided
    if save_path:
        results.save(filename=save_path)

    return detections
