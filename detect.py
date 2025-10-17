# detect.py

from ultralytics import YOLO
import os

# Load YOLOv8 model (it will download if not available)
model = YOLO("yolov8n.pt")  # You can use yolov8s.pt, yolov8m.pt, etc.

def detect_objects(image_path, output_path, target_class="chair"):
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

    # Save annotated image
    results.save(filename=output_path)

    return detections
