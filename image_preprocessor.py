"""
Image Preprocessor Module.

Provides:
- ImagePreprocessor: Class for adaptive histogram equalization (CLAHE)
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

ImageInputType = Union[str, Path, Image.Image, np.ndarray]


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

        raise ValueError(
            f"Unsupported image_input type: {type(image_input)}. "
            "Expected str, Path, PIL.Image.Image, or np.ndarray."
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
