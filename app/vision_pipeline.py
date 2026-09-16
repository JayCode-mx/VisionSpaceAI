"""Precision Furniture Object Detection Pipeline using YOLOv8 ONNX Runtime.

Optimized for high-precision furniture discovery under Render 512MB RAM constraints:
- Pure CPU ONNX Runtime session for YOLOv8 (yolov8n.onnx).
- Strict COCO Furniture Class Mapping:
    56: "chair",
    57: "couch",
    58: "potted plant",
    60: "dining table"
  (No generic/ambiguous names; strictly matches COCO labels).
- Class-Aware Non-Maximum Suppression (NMS) with configurable IoU threshold so
  overlapping objects (e.g., a dining table in front of a couch) are BOTH detected
  and never suppressed.
- Context-preserving `crop_objects` with 10-15% bounding box padding for downstream
  vector embedding models.
- Returns clean dictionaries containing crop, class label, and confidence score.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
from PIL import Image

try:
    from app.config import settings
except ImportError:
    from config import settings

logger = logging.getLogger("vision_pipeline")

# Strict COCO Dataset Furniture Mapping (DO NOT rename to 'sofa' or generic terms)
COCO_FURNITURE_CLASSES: Dict[int, str] = {
    56: "chair",
    57: "couch",
    58: "potted plant",
    60: "dining table",
}

# Backward-compatibility alias
TARGET_CLASSES = COCO_FURNITURE_CLASSES
CONF_THRESHOLD = 0.15
IOU_THRESHOLD = 0.45


def compute_iou(box1: Union[List[float], np.ndarray], box2: Union[List[float], np.ndarray]) -> float:
    """Calculate Intersection-over-Union (IoU) between two boxes [x1, y1, x2, y2]."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    if inter_area <= 0.0:
        return 0.0

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter_area

    return inter_area / union if union > 0 else 0.0


def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float = IOU_THRESHOLD,
) -> List[int]:
    """Perform Class-Aware Non-Maximum Suppression (NMS).

    CRITICAL: NMS is executed independently per class ID.
    This guarantees that overlapping objects of DIFFERENT classes (e.g. a
    'dining table' positioned in front of a 'couch') are BOTH retained and
    never suppress each other, while redundant duplicates of the SAME class
    are cleanly filtered out.

    Args:
        boxes: NumPy array of shape (N, 4) in [x1, y1, x2, y2] format.
        scores: NumPy array of shape (N,) with confidence scores.
        class_ids: NumPy array of shape (N,) with class IDs.
        iou_threshold: IoU overlap threshold for suppression (default 0.45).

    Returns:
        List of integer indices to keep, ordered by score descending.
    """
    if len(boxes) == 0:
        return []

    keep_indices: List[int] = []
    unique_classes = np.unique(class_ids)

    for cls in unique_classes:
        cls_mask = np.where(class_ids == cls)[0]
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]

        # Sort within this class by confidence descending
        order = np.argsort(-cls_scores)

        while order.size > 0:
            i = order[0]
            keep_indices.append(cls_mask[i])

            if order.size == 1:
                break

            # Calculate IoU between top box and remaining boxes in this class
            current_box = cls_boxes[i]
            remaining_boxes = cls_boxes[order[1:]]

            ious = np.array([compute_iou(current_box, b) for b in remaining_boxes])
            # Keep only boxes with IoU less than threshold
            remaining_mask = np.where(ious < iou_threshold)[0]
            order = order[remaining_mask + 1]

    # Return indices sorted by confidence score across all kept boxes
    return sorted(keep_indices, key=lambda idx: float(scores[idx]), reverse=True)


def crop_objects(
    image: Union[Image.Image, np.ndarray],
    boxes: List[Any],
    padding_percent: float = 0.12,
) -> List[Dict[str, Any]]:
    """Crop detected objects with 10-15% context padding.

    Adds contextual margin around each bounding box so visual embedding models
    (such as CLIP/ONNX models) capture full edges, contours, and texture context
    instead of aggressively cropped interiors.

    Args:
        image: Source PIL Image or NumPy array (RGB/BGR).
        boxes: List of detection dicts (containing 'box' or 'bbox', 'label', 'confidence')
               or raw box coordinates [x1, y1, x2, y2].
        padding_percent: Relative margin expansion per dimension (0.10 to 0.15).

    Returns:
        Clean list of dictionaries:
        [
            {
                "crop": <PIL.Image.Image or np.ndarray>,
                "class_label": "couch",
                "confidence": 0.89,
                "box": [x1, y1, x2, y2],
                "padded_box": [x1_pad, y1_pad, x2_pad, y2_pad],
            },
            ...
        ]
    """
    if isinstance(image, np.ndarray):
        img_h, img_w = image.shape[:2]
        is_numpy = True
    else:
        img_w, img_h = image.size
        is_numpy = False

    padding_percent = max(0.05, min(0.25, padding_percent))
    crops: List[Dict[str, Any]] = []

    for item in boxes:
        # Extract coordinates and metadata
        if isinstance(item, dict):
            raw_box = item.get("box", item.get("bbox", []))
            label = item.get("label", item.get("class_label", "furniture"))
            conf = float(item.get("confidence", item.get("score", 1.0)))
        elif isinstance(item, (list, tuple)):
            raw_box = item[:4]
            label = item[4] if len(item) > 4 else "furniture"
            conf = float(item[5]) if len(item) > 5 else 1.0
        else:
            continue

        if len(raw_box) < 4:
            continue

        x1, y1, x2, y2 = [float(v) for v in raw_box[:4]]
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)

        # 10-15% pixel expansion on each side
        pad_w = box_w * padding_percent
        pad_h = box_h * padding_percent

        x1_pad = max(0, int(round(x1 - pad_w)))
        y1_pad = max(0, int(round(y1 - pad_h)))
        x2_pad = min(img_w, int(round(x2 + pad_w)))
        y2_pad = min(img_h, int(round(y2 + pad_h)))

        # Guard against zero-area crops
        if x2_pad <= x1_pad or y2_pad <= y1_pad:
            continue

        if is_numpy:
            crop = image[y1_pad:y2_pad, x1_pad:x2_pad].copy()
        else:
            crop = image.crop((x1_pad, y1_pad, x2_pad, y2_pad))

        crops.append({
            "crop": crop,
            "class_label": str(label),
            "confidence": round(conf, 4),
            "box": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
            "padded_box": [x1_pad, y1_pad, x2_pad, y2_pad],
        })

    return crops


# Compatibility helper matching original function signature
def crop_with_padding(
    image: Union[Image.Image, np.ndarray],
    box: Union[List[int], Tuple[int, int, int, int]],
    padding_percent: float = 0.12,
) -> Union[Image.Image, np.ndarray]:
    """Backward compatibility helper to crop a single bounding box with padding."""
    res = crop_objects(image, [box], padding_percent=padding_percent)
    if res:
        return res[0]["crop"]
    # Fallback to direct slice
    if isinstance(image, np.ndarray):
        return image[box[1]:box[3], box[0]:box[2]]
    return image.crop((box[0], box[1], box[2], box[3]))


def filter_detections(
    boxes: Any,
    scores: Any,
    class_ids: Any,
    conf_threshold: float = CONF_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Filter raw detections by confidence and strict COCO furniture classes."""
    detected = []
    for box, score, class_id in zip(boxes, scores, class_ids):
        cid = int(class_id)
        sc = float(score)
        if sc >= conf_threshold and cid in COCO_FURNITURE_CLASSES:
            detected.append({
                "box": [int(round(b)) for b in box],
                "label": COCO_FURNITURE_CLASSES[cid],
                "confidence": round(sc, 4),
                "class_id": cid,
            })
    return detected


class VisionPipeline:
    """Lightweight YOLOv8 ONNX Object Detection & Feature Pipeline.

    Optimized for highest precision on furniture with strict COCO class mapping
    and memory under 512MB RAM.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or "cpu"
        self.target_classes = COCO_FURNITURE_CLASSES

        # CPU Session options for ultra-low memory
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1

        # 1. Locate YOLOv8 ONNX model
        yolo_path = self._find_model_path("yolov8n.onnx")
        if yolo_path:
            logger.info("Loading YOLOv8 ONNX model from %s...", yolo_path)
            self.yolo_session = ort.InferenceSession(
                str(yolo_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self.yolo_input_name = self.yolo_session.get_inputs()[0].name
            self.yolo_output_name = self.yolo_session.get_outputs()[0].name
        else:
            logger.warning("yolov8n.onnx not found. Detection session unavailable.")
            self.yolo_session = None

        # 2. Locate MobileNet/Feature model (optional for compatibility)
        feat_path = self._find_model_path("model.onnx")
        if feat_path:
            self.feat_session = ort.InferenceSession(
                str(feat_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
        else:
            self.feat_session = self.yolo_session

        logger.info("VisionPipeline initialized successfully.")

    def _find_model_path(self, filename: str) -> Optional[Path]:
        """Locate ONNX model across common project paths."""
        candidates = [
            Path("app/models") / filename,
            Path(__file__).parent / "models" / filename,
            Path(__file__).parent.parent / "models" / filename,
        ]
        for p in candidates:
            if p.is_file():
                return p
        return None

    @property
    def session(self):
        return self.yolo_session or self.feat_session

    @property
    def model(self):
        return self.session

    @property
    def mobilenet_model(self):
        return self.feat_session

    @property
    def clip_model(self):
        return self.session

    @property
    def clip_processor(self):
        return None

    def letterbox_image(
        self,
        image: Image.Image,
        target_size: int = 640,
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Resize image to target_size x target_size with aspect-preserving padding."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        orig_w, orig_h = image.size
        scale = min(target_size / orig_w, target_size / orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        pad_w = (target_size - new_w) // 2
        pad_h = (target_size - new_h) // 2

        canvas = Image.new("RGB", (target_size, target_size), (114, 114, 114))
        canvas.paste(resized, (pad_w, pad_h))

        # Float32 normalized to [0, 1] in shape (1, 3, 640, 640)
        img_arr = np.array(canvas, dtype=np.float32) / 255.0
        img_arr = np.transpose(img_arr, (2, 0, 1))
        tensor = np.expand_dims(img_arr, axis=0).astype(np.float32)

        return tensor, scale, (pad_w, pad_h)

    def detect(
        self,
        image: Union[Image.Image, np.ndarray],
        conf_threshold: float = CONF_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
    ) -> List[Dict[str, Any]]:
        """Detect furniture items in an image using YOLOv8 ONNX.

        Applies class-aware NMS to preserve overlapping objects of different
        classes (e.g., dining table in front of couch).

        Args:
            image: PIL Image or NumPy array.
            conf_threshold: Minimum confidence score (default: 0.15).
            iou_threshold: Non-maximum suppression IoU cutoff (default: 0.45).

        Returns:
            List of detected furniture dicts:
            [
                {
                    "box": [x1, y1, x2, y2],
                    "label": "couch",
                    "confidence": 0.88,
                    "class_id": 57
                },
                ...
            ]
        """
        if self.yolo_session is None:
            logger.warning("YOLOv8 session not loaded.")
            return []

        pil_image = image if isinstance(image, Image.Image) else Image.fromarray(image)
        orig_w, orig_h = pil_image.size

        # 1. Letterbox Preprocessing
        input_tensor, scale, (pad_w, pad_h) = self.letterbox_image(pil_image, target_size=640)

        # 2. ONNX Inference
        outputs = self.yolo_session.run([self.yolo_output_name], {self.yolo_input_name: input_tensor})
        raw_output = outputs[0]  # Shape: (1, 84, 8400)

        # 3. Transpose to (8400, 84): [cx, cy, w, h, score_0, ... score_79]
        preds = np.transpose(raw_output[0], (1, 0))  # Shape: (8400, 84)

        boxes_cxcywh = preds[:, :4]
        class_scores = preds[:, 4:]

        candidate_boxes = []
        candidate_scores = []
        candidate_class_ids = []

        # 4. Filter for strict COCO furniture classes (56: chair, 57: couch, 58: potted plant, 60: dining table)
        for class_id, class_name in COCO_FURNITURE_CLASSES.items():
            if class_id >= class_scores.shape[1]:
                continue

            scores_for_cls = class_scores[:, class_id]
            mask = scores_for_cls >= conf_threshold

            if not np.any(mask):
                continue

            valid_boxes = boxes_cxcywh[mask]
            valid_scores = scores_for_cls[mask]

            for (cx, cy, bw, bh), score in zip(valid_boxes, valid_scores):
                # Convert 640x640 letterbox coords to original image dimensions
                x1_canvas = cx - (bw / 2.0)
                y1_canvas = cy - (bh / 2.0)
                x2_canvas = cx + (bw / 2.0)
                y2_canvas = cy + (bh / 2.0)

                x1_orig = (x1_canvas - pad_w) / scale
                y1_orig = (y1_canvas - pad_h) / scale
                x2_orig = (x2_canvas - pad_w) / scale
                y2_orig = (y2_canvas - pad_h) / scale

                # Clamp to image boundaries
                x1_clamped = max(0, min(orig_w, int(round(x1_orig))))
                y1_clamped = max(0, min(orig_h, int(round(y1_orig))))
                x2_clamped = max(0, min(orig_w, int(round(x2_orig))))
                y2_clamped = max(0, min(orig_h, int(round(y2_orig))))

                if x2_clamped > x1_clamped and y2_clamped > y1_clamped:
                    candidate_boxes.append([x1_clamped, y1_clamped, x2_clamped, y2_clamped])
                    candidate_scores.append(float(score))
                    candidate_class_ids.append(class_id)

        if not candidate_boxes:
            return []

        # 5. Class-Aware Non-Maximum Suppression (Overlapping different classes survive!)
        kept_indices = non_max_suppression(
            np.array(candidate_boxes, dtype=np.float32),
            np.array(candidate_scores, dtype=np.float32),
            np.array(candidate_class_ids, dtype=np.int32),
            iou_threshold=iou_threshold,
        )

        detections: List[Dict[str, Any]] = []
        for idx in kept_indices:
            cid = candidate_class_ids[idx]
            detections.append({
                "box": candidate_boxes[idx],
                "label": COCO_FURNITURE_CLASSES[cid],
                "confidence": round(candidate_scores[idx], 4),
                "class_id": cid,
            })

        logger.info(
            "YOLOv8 detected %d furniture items: %s",
            len(detections),
            [f"{d['label']} ({d['confidence']})" for d in detections],
        )
        return detections

    def detect_and_crop(
        self,
        image: Union[Image.Image, np.ndarray],
        conf_threshold: float = CONF_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
        padding_percent: float = 0.12,
    ) -> List[Dict[str, Any]]:
        """End-to-end detection and context-aware cropping.

        Detects furniture items and extracts cropped regions with 10-15% padding.

        Returns:
            List of dicts:
            [
                {
                    "crop": <PIL.Image.Image or np.ndarray>,
                    "class_label": "couch",
                    "confidence": 0.88,
                    "box": [x1, y1, x2, y2],
                    "padded_box": [x1_pad, y1_pad, x2_pad, y2_pad],
                },
                ...
            ]
        """
        detections = self.detect(image, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
        return crop_objects(image, detections, padding_percent=padding_percent)

    def predict(self, image_np: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Run raw inference on session (backwards compatibility)."""
        if self.session is None:
            return np.zeros((1, 84, 8400), dtype=np.float32)
        if isinstance(image_np, Image.Image):
            tensor, _, _ = self.letterbox_image(image_np, target_size=640)
        else:
            tensor = image_np if image_np.ndim == 4 else np.expand_dims(image_np, axis=0)
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        return self.session.run([output_name], {input_name: tensor.astype(np.float32)})[0]
