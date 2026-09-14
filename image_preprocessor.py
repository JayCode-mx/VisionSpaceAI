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
)

__all__ = [
    "ImagePreprocessor",
    "compute_ssim_similarity",
    "ImageInputType",
    "crop_bounding_box",
    "ObjectDetector",
    "INTERIOR_CLASSES",
    "LABEL_TO_CATEGORY",
    "INTERIOR_COCO_IDS",
    "compute_bbox_iou",
    "extract_all_furniture_crops",
]
