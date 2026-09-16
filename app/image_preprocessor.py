"""Image Preprocessor Module for VisionSpace AI.

Provides:
- ImagePreprocessor: Adaptive histogram equalization (CLAHE)
  via OpenCV and aspect-ratio-preserving resize & pad to 224x224 via Pillow.
- compute_ssim_similarity: Utility function using scikit-image's structural_similarity
  to compute similarity scores between processed images.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.metrics import structural_similarity as skimage_ssim

ImageInputType = Union[str, Path, Image.Image, np.ndarray, bytes, bytearray]


class ImagePreprocessor:
    """Preprocesses images using OpenCV (CLAHE) and Pillow (aspect-ratio resize & pad).

    Attributes:
        target_size (Tuple[int, int]): Desired (width, height) output dimensions. Defaults to (224, 224).
        clip_limit (float): Threshold for contrast limiting in CLAHE. Defaults to 2.0.
        tile_grid_size (Tuple[int, int]): Size of grid for CLAHE histogram equalization. Defaults to (8, 8).
        pad_color (Tuple[int, int, int] | int): Fill color for padding (letterboxing). Defaults to (0, 0, 0).
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        pad_color: Union[Tuple[int, int, int], int] = (0, 0, 0),
    ) -> None:
        """Initialize ImagePreprocessor with configurable parameters.

        Args:
            target_size: Desired (width, height) output dimensions.
            clip_limit: Threshold for contrast limiting in CLAHE.
            tile_grid_size: Tile grid size for CLAHE (e.g. (8, 8)).
            pad_color: Padding color used when letterboxing (default black).
        """
        self.target_size = target_size
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.pad_color = pad_color
        self._clahe = cv2.createCLAHE(
            clipLimit=float(clip_limit),
            tileGridSize=tile_grid_size,
        )

    def load_image(self, image_input: ImageInputType) -> Image.Image:
        """Load or normalize an input image into a PIL Image.

        Args:
            image_input: File path (str or Path), PIL Image, or NumPy array (RGB/BGR/Grayscale).

        Returns:
            Image.Image: Loaded PIL Image.

        Raises:
            FileNotFoundError: If a given file path does not exist.
            ValueError: If input format or array shape is unsupported.
        """
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.is_file():
                raise FileNotFoundError(f"Image file not found: {path}")
            return Image.open(path).copy()

        if isinstance(image_input, Image.Image):
            return image_input.copy()

        if isinstance(image_input, np.ndarray):
            # Normalize dtype to uint8 if needed
            arr = image_input
            if arr.dtype != np.uint8:
                if np.issubdtype(arr.dtype, np.floating):
                    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
                else:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)

            if arr.ndim == 2:
                return Image.fromarray(arr, mode="L")
            elif arr.ndim == 3:
                channels = arr.shape[2]
                if channels == 1:
                    return Image.fromarray(arr.squeeze(-1), mode="L")
                elif channels == 3:
                    return Image.fromarray(arr, mode="RGB")
                elif channels == 4:
                    return Image.fromarray(arr, mode="RGBA")
                else:
                    raise ValueError(f"Unsupported number of channels in array: {channels}")
            else:
                raise ValueError(f"Unsupported array shape: {arr.shape}")

        if isinstance(image_input, (bytes, bytearray)):
            import io
            return Image.open(io.BytesIO(image_input)).copy()

        raise ValueError(
            f"Unsupported image_input type: {type(image_input)}. "
            "Expected str, Path, PIL.Image.Image, np.ndarray, or bytes."
        )

    def apply_clahe(self, image: Image.Image) -> Image.Image:
        """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) using OpenCV.

        For color images, the image is converted to LAB color space and CLAHE
        is applied to the Lightness (L) channel to preserve natural color tones.
        For grayscale images, CLAHE is applied directly.
        Alpha channels (RGBA) are preserved.

        Args:
            image: Input PIL Image.

        Returns:
            Image.Image: Contrast-enhanced PIL Image.
        """
        mode = image.mode
        alpha_channel = None

        if mode == "RGBA":
            # Split off alpha channel to avoid distorting transparency
            r, g, b, alpha_channel = image.split()
            rgb_image = Image.merge("RGB", (r, g, b))
            np_img = np.array(rgb_image)
        elif mode == "RGB":
            np_img = np.array(image)
        elif mode in ("L", "1"):
            np_img = np.array(image.convert("L"))
        else:
            # Fallback conversion to RGB
            np_img = np.array(image.convert("RGB"))
            mode = "RGB"

        # Apply CLAHE depending on color channels
        if np_img.ndim == 2:
            equalized = self._clahe.apply(np_img)
            result = Image.fromarray(equalized, mode="L")
        else:
            # Convert RGB to LAB, apply CLAHE on L-channel, then convert back to RGB
            lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            l_equalized = self._clahe.apply(l_channel)
            lab_equalized = cv2.merge((l_equalized, a_channel, b_channel))
            rgb_equalized = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2RGB)
            result = Image.fromarray(rgb_equalized, mode="RGB")

        if alpha_channel is not None:
            r, g, b = result.split()
            result = Image.merge("RGBA", (r, g, b, alpha_channel))

        return result

    def resize_and_pad(
        self,
        image: Image.Image,
        target_size: Optional[Tuple[int, int]] = None,
        pad_color: Optional[Union[Tuple[int, int, int], int]] = None,
    ) -> Image.Image:
        """Resize and pad the image using Pillow while preserving its aspect ratio.

        Centers the scaled image within a canvas of target_size (default 224x224).

        Args:
            image: PIL Image to resize and pad.
            target_size: Optional override for (width, height). Defaults to self.target_size.
            pad_color: Optional override for pad color. Defaults to self.pad_color.

        Returns:
            Image.Image: Resized and padded PIL Image of exactly target_size.
        """
        size = target_size or self.target_size
        color = pad_color if pad_color is not None else self.pad_color

        # Normalize pad_color for grayscale mode if necessary
        if image.mode in ("L", "1") and isinstance(color, tuple):
            # Use brightness / first channel if tuple provided for single-channel image
            color = color[0]

        return ImageOps.pad(
            image,
            size,
            method=Image.Resampling.LANCZOS,
            color=color,
            centering=(0.5, 0.5),
        )

    def crop_bounding_box(
        self,
        image_input: ImageInputType,
        bbox: Union[List[int], Tuple[int, int, int, int]],
    ) -> Image.Image:
        """Crop a sub-region defined by bounding box [x1, y1, x2, y2] from an image.

        Clips coordinates to image bounds and ensures valid positive dimensions.

        Args:
            image_input: Input PIL Image, numpy array, path, or bytes.
            bbox: [x1, y1, x2, y2] pixel coordinates.

        Returns:
            Image.Image: Cropped PIL Image.
        """
        img = self.load_image(image_input)
        w, h = img.size
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Clamp coordinates to image dimensions
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        return img.crop((x1, y1, x2, y2))

    def preprocess(
        self,
        image_input: ImageInputType,
        return_numpy: bool = False,
    ) -> Union[Image.Image, np.ndarray]:
        """Complete pipeline: loads image, applies CLAHE, resizes and pads to target size.

        Args:
            image_input: Image filepath (str/Path), PIL Image, or NumPy array.
            return_numpy: If True, returns a NumPy array; otherwise returns a PIL Image.

        Returns:
            Union[Image.Image, np.ndarray]: The preprocessed 224x224 image.
        """
        image = self.load_image(image_input)
        clahe_image = self.apply_clahe(image)
        processed_image = self.resize_and_pad(clahe_image)

        if return_numpy:
            return np.array(processed_image)
        return processed_image

    def __call__(
        self,
        image_input: ImageInputType,
        return_numpy: bool = False,
    ) -> Union[Image.Image, np.ndarray]:
        """Convenience method to allow calling the instance directly."""
        return self.preprocess(image_input, return_numpy=return_numpy)


def compute_ssim_similarity(
    image1: Union[ImageInputType, np.ndarray],
    image2: Union[ImageInputType, np.ndarray],
    channel_axis: Optional[int] = -1,
    data_range: Optional[float] = None,
    preprocessor: Optional[ImagePreprocessor] = None,
) -> float:
    """Compute the Structural Similarity Index (SSIM) between two images using scikit-image.

    Args:
        image1: First image (filepath, PIL Image, or NumPy array).
        image2: Second image (filepath, PIL Image, or NumPy array).
        channel_axis: Axis representing channels for multi-channel images (default -1).
            If None, images are treated as single-channel / grayscale.
        data_range: The dynamic range of the images. If None, 255.0 is used for uint8 images,
            or 1.0 for float images with values in [0, 1].
        preprocessor: Optional ImagePreprocessor instance. If provided, both images
            will first be preprocessed through it before computing SSIM.

    Returns:
        float: Structural similarity score between -1.0 and 1.0 (1.0 = identical).

    Raises:
        ValueError: If images have mismatched shapes or dimensions.
    """
    # Preprocess if requested
    if preprocessor is not None:
        arr1 = preprocessor.preprocess(image1, return_numpy=True)
        arr2 = preprocessor.preprocess(image2, return_numpy=True)
    else:
        def to_numpy(img: ImageInputType) -> np.ndarray:
            if isinstance(img, (str, Path)):
                return np.array(Image.open(img))
            elif isinstance(img, Image.Image):
                return np.array(img)
            elif isinstance(img, np.ndarray):
                return img
            raise ValueError(f"Unsupported image input type: {type(img)}")

        arr1 = to_numpy(image1)
        arr2 = to_numpy(image2)

    if arr1.shape != arr2.shape:
        raise ValueError(
            f"Image shapes must match to compute SSIM, got {arr1.shape} and {arr2.shape}. "
            "Pass preprocessor=ImagePreprocessor() to automatically align shapes to (224, 224)."
        )

    # Determine channel_axis
    if arr1.ndim == 2:
        effective_channel_axis = None
    elif arr1.ndim == 3:
        effective_channel_axis = channel_axis
    else:
        raise ValueError(f"Expected 2D or 3D image arrays, got shape {arr1.shape}")

    # Determine data_range
    if data_range is None:
        if np.issubdtype(arr1.dtype, np.integer):
            effective_data_range = 255.0
        elif np.issubdtype(arr1.dtype, np.floating):
            effective_data_range = 1.0 if arr1.max() <= 1.0 else 255.0
        else:
            effective_data_range = float(arr1.max() - arr1.min()) or 255.0
    else:
        effective_data_range = float(data_range)

    score = skimage_ssim(
        arr1,
        arr2,
        channel_axis=effective_channel_axis,
        data_range=effective_data_range,
    )
    return float(score)


def crop_bounding_box(
    image: Image.Image,
    bbox: Union[List[int], Tuple[int, int, int, int]],
) -> Image.Image:
    """Crop bounding box [x1, y1, x2, y2] from a PIL Image with boundary clamping."""
    w, h = image.size
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(x1 + 1, min(int(x2), w))
    y2 = max(y1 + 1, min(int(y2), h))
    return image.crop((x1, y1, x2, y2))


def crop_with_padding(
    image: Union[Image.Image, np.ndarray],
    box: Union[List[int], Tuple[int, int, int, int]],
    padding_percent: float = 0.10,
) -> Union[Image.Image, np.ndarray]:
    """Crop bounding box [x1, y1, x2, y2] with 10-15% contextual padding."""
    if isinstance(image, np.ndarray):
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box
        pad_w = (x2 - x1) * padding_percent
        pad_h = (y2 - y1) * padding_percent
        x1_pad = max(0, int(x1 - pad_w))
        y1_pad = max(0, int(y1 - pad_h))
        x2_pad = min(w, int(x2 + pad_w))
        y2_pad = min(h, int(y2 + pad_h))
        return image[y1_pad:y2_pad, x1_pad:x2_pad]
    else:
        w, h = image.size
        x1, y1, x2, y2 = box
        pad_w = (x2 - x1) * padding_percent
        pad_h = (y2 - y1) * padding_percent
        x1_pad = max(0, int(x1 - pad_w))
        y1_pad = max(0, int(y1 - pad_h))
        x2_pad = min(w, int(x2 + pad_w))
        y2_pad = min(h, int(y2 + pad_h))
        return image.crop((x1_pad, y1_pad, x2_pad, y2_pad))


# Re-export YOLOv8 ObjectDetector and interior classes for convenience
try:
    from app.object_detector import (
        ObjectDetector,
        INTERIOR_CLASSES,
        LABEL_TO_CATEGORY,
        INTERIOR_COCO_IDS,
        compute_bbox_iou,
        TARGET_CLASSES,
        CONF_THRESHOLD,
        filter_detections,
    )
except ImportError:
    try:
        from object_detector import (
            ObjectDetector,
            INTERIOR_CLASSES,
            LABEL_TO_CATEGORY,
            INTERIOR_COCO_IDS,
            compute_bbox_iou,
            TARGET_CLASSES,
            CONF_THRESHOLD,
            filter_detections,
        )
    except ImportError:
        ObjectDetector = None
        INTERIOR_CLASSES = {}
        LABEL_TO_CATEGORY = {}
        INTERIOR_COCO_IDS = {}
        compute_bbox_iou = None
        TARGET_CLASSES = {56: "Chair", 57: "Sofa", 58: "Plant", 60: "Table"}
        CONF_THRESHOLD = 0.18
        filter_detections = None


_default_detector = None


def get_default_detector():
    """Lazily instantiate or return shared ObjectDetector instance."""
    global _default_detector
    if _default_detector is None and ObjectDetector is not None:
        try:
            _default_detector = ObjectDetector(model_name="yolov8n.pt", confidence_threshold=0.18, iou_threshold=0.45)
        except Exception:
            _default_detector = None
    return _default_detector


def extract_all_furniture_crops(
    image: Image.Image,
    detector: Optional[Any] = None,
    max_crops: int = 6,
) -> List[dict]:
    """Hybrid Region Extractor combining YOLOv8 and Heuristic Region Sampling.

    1. Primary Stage (YOLOv8):
       Run YOLO detection to identify explicit bounding boxes (sofas, chairs, tables, TV units).
    2. Coverage Check:
       Check if a table/surface object was detected in the lower-center region
       (Y: 45%-85%, X: 25%-75%).
    3. Heuristic Fallback:
       If no table box is detected in that region, automatically generate a candidate
       bounding box for the lower-center quadrant (where living room coffee tables sit).
    4. Crop all generated bounding boxes (YOLO boxes + Heuristic Coffee Table box).

    Args:
        image: Original PIL Image.
        detector: Optional ObjectDetector instance. If None, default detector is loaded.
        max_crops: Maximum number of furniture crops to extract (default: 4).

    Returns:
        List of crop dictionaries with bounding boxes and cropped PIL images.
    """
    w, h = image.size
    crops: List[dict] = []

    # 1. Primary Stage (YOLOv8)
    if detector is None:
        detector = get_default_detector()

    detected_raw = []
    if detector is not None:
        try:
            detected_raw = detector.detect(image)
            print(f"\n🎯 [extract_all_furniture_crops] YOLO detector returned {len(detected_raw)} items")
            for d in detected_raw:
                print(f"   • {d.get('label')} (conf={d.get('confidence')}, bbox={d.get('bbox')})")
        except Exception as exc:
            print(f"Warning: YOLO detector failed ({exc}), falling back to heuristic crops.")
            detected_raw = []
    else:
        print(f"⚠️ [extract_all_furniture_crops] No detector provided, using heuristic only")

    # 2. Coverage Check: Check if table/surface object is detected in lower-center region
    # Lower-center target region: Y: 45%-85%, X: 25%-75%
    lc_x1 = int(0.25 * w)
    lc_y1 = int(0.45 * h)
    lc_x2 = int(0.75 * w)
    lc_y2 = int(0.85 * h)
    lc_box = [lc_x1, lc_y1, lc_x2, lc_y2]

    table_in_lower_center = False
    for det in detected_raw:
        label = str(det.get("label", "")).lower()
        cat = str(det.get("category_hint", "")).lower()
        bbox = det.get("bbox", [0, 0, 0, 0])

        is_table = any(t in label for t in ["table", "desk", "coffee table"]) or "table" in cat
        if is_table:
            # Check overlap or center point in lower-center zone
            overlap = compute_bbox_iou(bbox, lc_box) if compute_bbox_iou else 0.0
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            if overlap > 0.15 or (lc_x1 <= cx <= lc_x2 and lc_y1 <= cy <= lc_y2):
                table_in_lower_center = True
                break

    # Add valid YOLO detections
    for det in detected_raw:
        bbox = det["bbox"]
        cropped = det.get("cropped_image")
        if cropped is None:
            cropped = crop_bounding_box(image, bbox)

        crops.append({
            "item_id": len(crops) + 1,
            "label": det.get("label", "furniture"),
            "category": det.get("category_hint", "Furniture"),
            "confidence": round(float(det.get("confidence", 0.85)), 4),
            "bbox": bbox,
            "cropped_image": cropped,
            "is_heuristic": False,
        })

    # 3. Heuristic Fallback: If no table box is detected in lower-center region,
    # automatically generate a candidate bounding box for the lower-center quadrant
    if not table_in_lower_center and w >= 80 and h >= 80:
        # Check if an existing YOLO box heavily overlaps the lower-center quadrant
        heavy_overlap = False
        for c in crops:
            if compute_bbox_iou and compute_bbox_iou(c["bbox"], lc_box) > 0.65:
                heavy_overlap = True
                break

        if not heavy_overlap:
            heuristic_crop = crop_bounding_box(image, lc_box)
            crops.append({
                "item_id": len(crops) + 1,
                "label": "coffee table",
                "category": "Table",
                "confidence": 0.55,
                "bbox": lc_box,
                "cropped_image": heuristic_crop,
                "is_heuristic": True,
            })

    # Fallback if no crops detected at all: full image
    if len(crops) == 0:
        crops.append({
            "item_id": 1,
            "label": "furniture",
            "category": "Furniture",
            "confidence": 1.0,
            "bbox": [0, 0, w, h],
            "cropped_image": image.copy(),
            "is_heuristic": False,
        })

    # Limit to max_crops
    crops = crops[:max_crops]

    # Re-index item_ids sequentially
    for idx, c in enumerate(crops, start=1):
        c["item_id"] = idx

    print(f"\n📦 [extract_all_furniture_crops] FINAL: Returning {len(crops)} crops (max_crops={max_crops})")
    for c in crops:
        print(f"   • #{c['item_id']} {c.get('label')} ({c.get('category')}) conf={c.get('confidence')} heuristic={c.get('is_heuristic')}")

    return crops
