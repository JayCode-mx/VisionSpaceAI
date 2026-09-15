"""Root alias re-exporting ImagePreprocessor and compute_ssim_similarity from app.image_preprocessor."""

from app.image_preprocessor import (
    ImagePreprocessor,
    compute_ssim_similarity,
    ImageInputType,
    crop_bounding_box,
    ObjectDetector,
    INTERIOR_CLASSES,
    LABEL_TO_CATEGORY,
    INTERIOR_COCO_IDS,
    compute_bbox_iou,
    extract_all_furniture_crops,
    crop_with_padding,
    filter_detections,
    TARGET_CLASSES,
    CONF_THRESHOLD,
)

__all__ = [
    "ImagePreprocessor",
    "compute_ssim_similarity",
    "ImageInputType",
    "crop_bounding_box",
    "crop_with_padding",
    "ObjectDetector",
    "INTERIOR_CLASSES",
    "LABEL_TO_CATEGORY",
    "INTERIOR_COCO_IDS",
    "compute_bbox_iou",
    "extract_all_furniture_crops",
    "filter_detections",
    "TARGET_CLASSES",
    "CONF_THRESHOLD",
]
