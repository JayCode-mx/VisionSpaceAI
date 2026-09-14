"""Object Detection Pipeline using Ultralytics YOLOv8 for VisionSpace AI.

Detects interior and furniture objects in room images, extracts cropped regions,
and standardizes class labels for downstream CLIP embedding and vector retrieval.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# Interior and furniture target classes to detect
INTERIOR_CLASSES = {
    "chair": "chair",
    "couch": "sofa",
    "sofa": "sofa",
    "potted plant": "potted plant",
    "bed": "bed",
    "dining table": "dining table",
    "tv": "tv",
    "laptop": "laptop",
    "clock": "clock",
    "vase": "vase",
}

# Mapping from detected class labels to Qdrant catalog categories
LABEL_TO_CATEGORY = {
    "sofa": "Sofa",
    "couch": "Sofa",
    "chair": "Chair",
    "dining table": "Table",
    "bed": "Bed",
}


class ObjectDetector:
    """YOLOv8-based object detection and bounding box extraction for interior furniture."""

    def __init__(self, model_name: str = "yolov8n.pt", confidence_threshold: float = 0.25):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load YOLOv8 model weights."""
        if YOLO is None:
            print("Warning: 'ultralytics' not installed. Object detection will operate in fallback mode.")
            return

        try:
            print(f"Loading YOLOv8 object detector ({self.model_name})...")
            self.model = YOLO(self.model_name)
            print("YOLOv8 object detector successfully loaded.")
        except Exception as exc:
            print(f"Warning: Failed to load YOLOv8 model ({exc}). Fallback mode active.")
            self.model = None

    def detect(
        self,
        image: Image.Image,
        confidence: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Run object detection on a PIL Image and filter for interior/furniture classes.

        Args:
            image: Original PIL Image.
            confidence: Optional confidence threshold override.

        Returns:
            List[Dict[str, Any]]: List of detected objects:
                [
                    {
                        "object_id": 1,
                        "label": "sofa",
                        "category_hint": "Sofa",
                        "confidence": 0.91,
                        "bbox": [x1, y1, x2, y2],
                        "cropped_image": <PIL.Image>,
                    }, ...
                ]
        """
        conf_thresh = confidence if confidence is not None else self.confidence_threshold
        w, h = image.size

        if self.model is None:
            # Fallback when YOLO is not loaded
            return []

        try:
            # Run inference on PIL Image
            results = self.model(image, conf=conf_thresh, verbose=False)
            if not results or len(results) == 0:
                return []

            result = results[0]
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                return []

            detections: List[Dict[str, Any]] = []
            object_idx = 1

            for box in boxes:
                cls_id = int(box.cls[0].item())
                raw_label = result.names.get(cls_id, "").lower().strip()
                score = float(box.conf[0].item())

                # Check if detected class is within our interior/furniture targets
                matched_label = None
                for target_key, canonical_name in INTERIOR_CLASSES.items():
                    if raw_label == target_key or target_key in raw_label:
                        matched_label = canonical_name
                        break

                if matched_label is None:
                    continue

                # Get bounding box coordinates [x1, y1, x2, y2]
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                x1, y1, x2, y2 = xyxy

                # Clamp to image boundaries
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(x1 + 1, min(x2, w))
                y2 = max(y1 + 1, min(y2, h))

                # Crop bounding box area from main image
                crop_img = image.crop((x1, y1, x2, y2))

                detections.append({
                    "object_id": object_idx,
                    "label": matched_label,
                    "category_hint": LABEL_TO_CATEGORY.get(matched_label),
                    "confidence": round(score, 4),
                    "bbox": [x1, y1, x2, y2],
                    "cropped_image": crop_img,
                })
                object_idx += 1

            # Sort detections by bounding box area descending (largest/primary object first)
            detections.sort(
                key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
                reverse=True,
            )
            for idx, det in enumerate(detections, start=1):
                det["object_id"] = idx

            return detections

        except Exception as exc:
            print(f"Error during YOLOv8 detection inference: {exc}")
            return []
