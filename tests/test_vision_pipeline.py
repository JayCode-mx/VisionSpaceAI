"""Unit tests for the upgraded YOLOv8 ONNX VisionPipeline."""

import numpy as np
from PIL import Image
import pytest

from app.vision_pipeline import (
    COCO_FURNITURE_CLASSES,
    VisionPipeline,
    compute_iou,
    crop_objects,
    non_max_suppression,
)


def test_coco_classes_exact_mapping():
    """Verify strict COCO classes without generic substitutions."""
    assert 56 in COCO_FURNITURE_CLASSES and COCO_FURNITURE_CLASSES[56] == "chair"
    assert 57 in COCO_FURNITURE_CLASSES and COCO_FURNITURE_CLASSES[57] == "couch"
    assert 58 in COCO_FURNITURE_CLASSES and COCO_FURNITURE_CLASSES[58] == "potted plant"
    assert 60 in COCO_FURNITURE_CLASSES and COCO_FURNITURE_CLASSES[60] == "dining table"
    assert "sofa" not in COCO_FURNITURE_CLASSES.values()


def test_nms_overlapping_objects_retained():
    """Verify a table in front of a couch is not suppressed by NMS."""
    # Couch box and overlapping table box
    boxes = np.array([
        [50, 50, 400, 350],   # Couch (class 57)
        [100, 150, 350, 350],  # Dining Table (class 60) overlapping
        [52, 53, 398, 348],   # Duplicate Couch (class 57)
    ], dtype=np.float32)
    scores = np.array([0.92, 0.88, 0.72], dtype=np.float32)
    class_ids = np.array([57, 60, 57], dtype=np.int32)

    kept = non_max_suppression(boxes, scores, class_ids, iou_threshold=0.45)
    assert 0 in kept  # Couch kept
    assert 1 in kept  # Dining table kept
    assert 2 not in kept  # Duplicate couch suppressed


def test_crop_objects_padding():
    """Verify crop_objects adds 10-15% pixel padding and returns clean dicts."""
    img = Image.new("RGB", (500, 500), color=(200, 200, 200))
    detections = [
        {"box": [100, 100, 200, 200], "label": "couch", "confidence": 0.85},
        {"box": [250, 250, 350, 350], "label": "dining table", "confidence": 0.90},
    ]

    crops = crop_objects(img, detections, padding_percent=0.12)
    assert len(crops) == 2

    for c in crops:
        assert "crop" in c
        assert "class_label" in c
        assert "confidence" in c
        assert c["class_label"] in ["couch", "dining table"]

        # 100x100 box with 12% padding on each side -> (100 + 24) = 124
        assert c["crop"].size[0] >= 120
        assert c["crop"].size[1] >= 120
