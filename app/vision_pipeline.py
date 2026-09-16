"""Precision Furniture Object Detection & Context-Aware Cropping Pipeline using YOLOv8 ONNX.

Optimized for high-precision furniture visual search via Google Lens (SerpApi)
while strictly adhering to Render's 512MB RAM free tier limit:
- Pure CPU ONNX Runtime session for YOLOv8 (yolov8n.onnx) with single-threaded ops (< 150MB RAM).
- Specific COCO Furniture Class Mapping without generic substitutions:
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table"
- Class-Aware Non-Maximum Suppression (NMS) with non-aggressive IoU threshold (0.50) so
  overlapping items (e.g., dining table in front of a sofa, or chairs around a table) are
  distinctly preserved without aggressive suppression.
- Context-preserving `crop_objects` with 15% pixel padding around detected bounding boxes
  to feed rich visual context (legs, contours, material, room setting) into Google Lens.
- Returns clean list of cropped PIL images, specific class labels, bounding boxes, and YOLO confidence scores.
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

# Strict COCO Dataset Furniture Mapping (Specific labels; no generic 'furniture' replacements)
COCO_FURNITURE_CLASSES: Dict[int, str] = {
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
}

# Backward-compatibility alias
TARGET_CLASSES = COCO_FURNITURE_CLASSES
CONF_THRESHOLD = 0.15
IOU_THRESHOLD = 0.50
PADDING_PERCENT = 0.15  # 15% context padding for Google Lens


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

    CRITICAL FOR E-COMMERCE:
    - Overlapping items of DIFFERENT classes (e.g., dining table in front of couch,
      or chair tucked into table) never suppress each other.
    - Multiple distinct items of the SAME class (e.g., 3 separate chairs) are ALL retained
      as long as they do not heavily overlap each other (IoU <= iou_threshold).

    Args:
        boxes: NumPy array of shape (N, 4) in [x1, y1, x2, y2] format.
        scores: NumPy array of shape (N,) with confidence scores.
        class_ids: NumPy array of shape (N,) with class IDs.
        iou_threshold: IoU overlap threshold for suppression (default 0.50).

    Returns:
        List of integer indices to keep, ordered by confidence score descending.
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

            current_box = cls_boxes[i]
            remaining_boxes = cls_boxes[order[1:]]

            ious = np.array([compute_iou(current_box, b) for b in remaining_boxes])
            # Keep boxes that don't heavily overlap with the current top detection
            remaining_mask = np.where(ious < iou_threshold)[0]
            order = order[remaining_mask + 1]

    # Return indices sorted by confidence score across all kept boxes
    return sorted(keep_indices, key=lambda idx: float(scores[idx]), reverse=True)


def crop_objects(
    image: Union[Image.Image, np.ndarray],
    boxes: List[Any],
    padding_percent: float = PADDING_PERCENT,
) -> List[Dict[str, Any]]:
    """Crop detected objects with 15% context padding.

    Adds contextual pixel margin around each bounding box so Google Lens receives
    the visual cues (chair legs, surroundings, fabric texture, perspective) required
    to accurately differentiate product types (e.g. lawn chair vs. dining chair).

    Args:
        image: Source PIL Image or NumPy array.
        boxes: List of detection dicts (with 'box' or 'bbox', 'label', 'confidence')
               or raw coordinate tuples [x1, y1, x2, y2].
        padding_percent: Relative margin expansion per dimension (default 0.15 / 15%).

    Returns:
        Clean list of dictionaries:
        [
            {
                "crop": <PIL.Image.Image in RGB mode>,
                "class_label": "chair",
                "confidence": 0.8924,
                "box": [x1, y1, x2, y2],
                "padded_box": [x1_pad, y1_pad, x2_pad, y2_pad],
            },
            ...
        ]
    """
    # Normalize input image to RGB PIL Image
    if isinstance(image, np.ndarray):
        img_h, img_w = image.shape[:2]
        if image.ndim == 3 and image.shape[2] == 3:
            # OpenCV BGR to RGB
            pil_source = Image.fromarray(image[..., ::-1])
        else:
            pil_source = Image.fromarray(image).convert("RGB")
    elif isinstance(image, Image.Image):
        pil_source = image.convert("RGB") if image.mode != "RGB" else image.copy()
        img_w, img_h = pil_source.size
    else:
        logger.error("Unsupported image type for cropping: %s", type(image))
        return []

    padding_percent = max(0.0, min(0.35, padding_percent))
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

        # 15% pixel expansion on each side
        pad_w = box_w * padding_percent
        pad_h = box_h * padding_percent

        x1_pad = max(0, int(round(x1 - pad_w)))
        y1_pad = max(0, int(round(y1 - pad_h)))
        x2_pad = min(img_w, int(round(x2 + pad_w)))
        y2_pad = min(img_h, int(round(y2 + pad_h)))

        # Guard against zero-area crops
        if x2_pad <= x1_pad or y2_pad <= y1_pad:
            continue

        # Extract context-padded crop as RGB PIL Image
        crop = pil_source.crop((x1_pad, y1_pad, x2_pad, y2_pad))

        crops.append({
            "crop": crop,
            "class_label": str(label),
            "confidence": round(conf, 4),
            "box": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
            "padded_box": [x1_pad, y1_pad, x2_pad, y2_pad],
        })

    return crops


def crop_with_padding(
    image: Union[Image.Image, np.ndarray],
    box: Union[List[int], Tuple[int, int, int, int]],
    padding_percent: float = PADDING_PERCENT,
) -> Image.Image:
    """Helper to crop a single bounding box with 15% padding."""
    res = crop_objects(image, [box], padding_percent=padding_percent)
    if res:
        return res[0]["crop"]
    if isinstance(image, Image.Image):
        return image.crop((box[0], box[1], box[2], box[3]))
    return Image.fromarray(image[box[1]:box[3], box[0]:box[2]]).convert("RGB")


def filter_detections(
    boxes: Any,
    scores: Any,
    class_ids: Any,
    conf_threshold: float = CONF_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Filter raw detections by confidence and specific COCO furniture classes."""
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
    """Lightweight YOLOv8 ONNX Object Detection & Contextual Cropping Pipeline.

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

        # Locate YOLOv8 ONNX model
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

        # Feature model compatibility session
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
        """Detect all furniture items in an image using YOLOv8 ONNX.

        Applies class-aware NMS to preserve overlapping objects of different classes
        and collects ALL detected bounding boxes (e.g. 3 distinct chairs are all preserved).

        Args:
            image: PIL Image or NumPy array.
            conf_threshold: Minimum confidence score (default: 0.15).
            iou_threshold: Non-maximum suppression IoU cutoff (default: 0.50).

        Returns:
            List of detected furniture dicts:
            [
                {
                    "box": [x1, y1, x2, y2],
                    "label": "chair",
                    "confidence": 0.88,
                    "class_id": 56
                },
                ...
            ]
        """
        if self.yolo_session is None:
            logger.warning("YOLOv8 session not loaded.")
            return []

        pil_image = image if isinstance(image, Image.Image) else Image.fromarray(image)
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
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

        # 4. Extract all bounding boxes matching specific COCO furniture classes
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
                # Convert 640x640 letterbox coords back to original image dimensions
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

        # 5. Class-Aware Non-Maximum Suppression (Overlapping different classes survive;
        #    multiple distinct items of same class are preserved)
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
        padding_percent: float = PADDING_PERCENT,
    ) -> List[Dict[str, Any]]:
        """End-to-end detection and context-aware 15% padding cropping.

        Detects all distinct furniture items and extracts cropped regions with 15% padding
        ready for high-precision Google Lens visual searches.

        Returns:
            List of dicts:
            [
                {
                    "crop": <PIL.Image.Image in RGB mode>,
                    "class_label": "chair",
                    "confidence": 0.88,
                    "box": [x1, y1, x2, y2],
                    "padded_box": [x1_pad, y1_pad, x2_pad, y2_pad],
                },
                ...
            ]
        """
        detections = self.detect(image, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
        return crop_objects(image, detections, padding_percent=padding_percent)

    # Aliases for backward and multi-pipeline compatibility
    def extract_all_furniture_crops(
        self,
        image: Union[Image.Image, np.ndarray],
        conf_threshold: float = CONF_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
        padding_percent: float = PADDING_PERCENT,
    ) -> List[Dict[str, Any]]:
        """Alias for detect_and_crop."""
        return self.detect_and_crop(
            image=image,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            padding_percent=padding_percent,
        )

    def extract_crops(
        self,
        image: Union[Image.Image, np.ndarray],
        conf_threshold: float = CONF_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
        padding_percent: float = PADDING_PERCENT,
    ) -> List[Dict[str, Any]]:
        """Alias for detect_and_crop."""
        return self.detect_and_crop(
            image=image,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            padding_percent=padding_percent,
        )

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
