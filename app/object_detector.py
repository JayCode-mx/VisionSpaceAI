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

# Interior and furniture target classes to detect (including COCO classes 56, 57, 58, 59, 60, 62, 74, 75)
INTERIOR_CLASSES = {
    "chair": "chair",
    "couch": "sofa",
    "sofa": "sofa",
    "dining table": "dining table",
    "table": "dining table",
    "coffee table": "dining table",
    "desk": "dining table",
    "tv": "tv",
    "tv monitor": "tv",
    "potted plant": "potted plant",
    "plant": "potted plant",
    "vase": "vase",
    "clock": "clock",
    "bed": "bed",
    "bench": "chair",
}

# Mapping from detected class labels to catalog categories
LABEL_TO_CATEGORY = {
    "sofa": "Sofa",
    "chair": "Chair",
    "dining table": "Table",
    "tv": "Living Room",
    "potted plant": "Decor",
    "vase": "Decor",
    "clock": "Decor",
    "bed": "Bed",
}

# Explicit COCO class IDs for interior & living room furniture
INTERIOR_COCO_IDS = {
    56: "chair",
    57: "sofa",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    62: "tv",
    74: "clock",
    75: "vase",
}


def compute_bbox_iou(boxA: List[int], boxB: List[int]) -> float:
    """Calculate Intersection-over-Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = float(areaA + areaB - inter_area)

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


class ObjectDetector:
    """YOLOv8-based object detection and bounding box extraction for interior furniture."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.20,
        iou_threshold: float = 0.45,
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
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

    def _parse_boxes(
        self,
        boxes,
        names: Dict[int, str],
        image: Image.Image,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> List[Dict[str, Any]]:
        """Parse YOLO detection boxes into structured object dictionaries."""
        w, h = image.size
        detections = []

        for box in boxes:
            cls_id = int(box.cls[0].item())
            score = float(box.conf[0].item())
            raw_label = names.get(cls_id, "").lower().strip()

            # Match label against known interior classes or COCO interior IDs
            matched_label = None
            if cls_id in INTERIOR_COCO_IDS:
                matched_label = INTERIOR_COCO_IDS[cls_id]
            else:
                for target_key, canonical_name in INTERIOR_CLASSES.items():
                    if raw_label == target_key or target_key in raw_label:
                        matched_label = canonical_name
                        break

            if matched_label is None:
                continue

            # Bounding box coordinates
            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
            x1 = xyxy[0] + offset_x
            y1 = xyxy[1] + offset_y
            x2 = xyxy[2] + offset_x
            y2 = xyxy[3] + offset_y

            # Clamp coordinates to full image bounds
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            crop_img = image.crop((x1, y1, x2, y2))

            detections.append({
                "label": matched_label,
                "category_hint": LABEL_TO_CATEGORY.get(matched_label, "Furniture"),
                "confidence": round(score, 4),
                "bbox": [x1, y1, x2, y2],
                "cropped_image": crop_img,
            })

        return detections

    def detect(
        self,
        image: Image.Image,
        confidence: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Run object detection on a PIL Image with sliced center-crop detection fallback.

        Args:
            image: Original PIL Image.
            confidence: Optional confidence threshold override (default: 0.20).
            iou: Optional IoU threshold override (default: 0.45).

        Returns:
            List[Dict[str, Any]]: List of detected objects with bounding boxes and crops.
        """
        conf_thresh = confidence if confidence is not None else self.confidence_threshold
        iou_thresh = iou if iou is not None else self.iou_threshold
        w, h = image.size

        if self.model is None:
            return []

        try:
            # 1. Full Image Inference with conf=0.20 and iou=0.45
            results = self.model(
                image,
                conf=conf_thresh,
                iou=iou_thresh,
                verbose=False,
            )

            detections: List[Dict[str, Any]] = []
            if results and len(results) > 0 and results[0].boxes is not None:
                detections = self._parse_boxes(
                    results[0].boxes,
                    results[0].names,
                    image,
                    offset_x=0,
                    offset_y=0,
                )

            # 2. Sliced / Center-Crop Detection Fallback
            # If YOLO detects <= 1 item in a multi-furniture room photo, run inference
            # on the center-crop region where coffee tables and centerpieces sit.
            if len(detections) <= 1 and w >= 80 and h >= 80:
                cx1 = int(0.12 * w)
                cy1 = int(0.30 * h)
                cx2 = int(0.88 * w)
                cy2 = int(0.90 * h)

                center_slice = image.crop((cx1, cy1, cx2, cy2))
                slice_results = self.model(
                    center_slice,
                    conf=max(0.15, conf_thresh - 0.03),
                    iou=iou_thresh,
                    verbose=False,
                )

                if slice_results and len(slice_results) > 0 and slice_results[0].boxes is not None:
                    slice_detections = self._parse_boxes(
                        slice_results[0].boxes,
                        slice_results[0].names,
                        image,
                        offset_x=cx1,
                        offset_y=cy1,
                    )

                    # Add non-duplicate detections from sliced inference
                    for s_det in slice_detections:
                        is_duplicate = False
                        for existing in detections:
                            if compute_bbox_iou(s_det["bbox"], existing["bbox"]) > 0.45:
                                is_duplicate = True
                                break
                        if not is_duplicate:
                            detections.append(s_det)

                # If still only 1 item detected and center area has low overlap (< 0.65) with first item,
                # extract center crop as coffee table / centerpiece candidate
                if len(detections) == 1:
                    primary_box = detections[0]["bbox"]
                    center_box = [cx1, cy1, cx2, cy2]
                    overlap = compute_bbox_iou(primary_box, center_box)
                    if overlap < 0.65:
                        center_crop = image.crop((cx1, cy1, cx2, cy2))
                        detections.append({
                            "label": "dining table",
                            "category_hint": "Table",
                            "confidence": 0.65,
                            "bbox": center_box,
                            "cropped_image": center_crop,
                        })

            # 3. Sort detections by bounding box area descending
            detections.sort(
                key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
                reverse=True,
            )

            # Assign sequential object_ids
            for idx, det in enumerate(detections, start=1):
                det["object_id"] = idx

            return detections

        except Exception as exc:
            print(f"Error during YOLOv8 detection inference: {exc}")
            return []
