from ultralytics import YOLO
import base64
import numpy as np
import cv2
from PIL import Image
import io
import os
import logging

logger = logging.getLogger(__name__)
detection_model = None  # Global YOLO model object

def load_detection_model():
    global detection_model
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'yolov8s_custom.pt')  # ✅ Correct model name
        detection_model = YOLO(model_path)
        logger.info("YOLOv8 model loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to load YOLOv8 model: {str(e)}")
        return False

def run_object_detection_with_boxes(image_data):
    try:
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)

        # Run YOLOv8 inference
        results = detection_model.predict(source=image_np, save=False, conf=0.25)

        if not results or not results[0].boxes:
            return {
                "success": True,
                "detections": [],
                "annotated_image": encode_image(image_np)
            }

        boxes = results[0].boxes
        class_names = detection_model.names
        annotated_img = image_np.copy()

        detections = []
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{class_names[cls]} ({conf:.2f})"

            # Draw bounding box
            cv2.rectangle(annotated_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)

            # Safe label Y position
            label_y = xyxy[1] - 10 if xyxy[1] - 10 > 10 else xyxy[1] + 20

            # Draw label
            cv2.putText(
                annotated_img,
                label,
                (xyxy[0], label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,           # Font scale
                (0, 0, 255),   # blue color
                3             # Thickness
            )

            detections.append({
                "xmin": int(xyxy[0]),
                "ymin": int(xyxy[1]),
                "xmax": int(xyxy[2]),
                "ymax": int(xyxy[3]),
                "confidence": conf,
                "class": cls,
                "name": class_names[cls]
            })

        return {
            "success": True,
            "detections": detections,
            "annotated_image": encode_image(annotated_img)
        }

    except Exception as e:
        logger.error(f"Error during YOLOv8 detection: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def encode_image(image_np):
    _, buffer = cv2.imencode('.jpg', image_np)
    base64_image = base64.b64encode(buffer).decode("utf-8")
    return base64_image
