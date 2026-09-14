"""Unit tests for image_preprocessor module."""

import unittest
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image

from image_preprocessor import ImagePreprocessor, compute_ssim_similarity


class TestImagePreprocessor(unittest.TestCase):

    def setUp(self):
        self.preprocessor = ImagePreprocessor(
            target_size=(224, 224),
            clip_limit=2.0,
            tile_grid_size=(8, 8),
            pad_color=(0, 0, 0),
        )

    def test_initialization(self):
        self.assertEqual(self.preprocessor.target_size, (224, 224))
        self.assertEqual(self.preprocessor.clip_limit, 2.0)
        self.assertEqual(self.preprocessor.tile_grid_size, (8, 8))
        self.assertEqual(self.preprocessor.pad_color, (0, 0, 0))

    def test_landscape_image_resize_and_pad(self):
        # 400x200 landscape image (2:1 aspect ratio)
        img = Image.new("RGB", (400, 200), color=(120, 80, 50))
        processed = self.preprocessor.preprocess(img)

        self.assertIsInstance(processed, Image.Image)
        self.assertEqual(processed.size, (224, 224))

        # Expected scaled size: width=224, height=112 (padded top and bottom: 56px each)
        arr = np.array(processed)
        # Top padding should be black (0, 0, 0)
        np.testing.assert_array_equal(arr[0, 112], [0, 0, 0])
        np.testing.assert_array_equal(arr[50, 112], [0, 0, 0])
        # Center should contain equalized pixels (not pure black)
        self.assertTrue(np.any(arr[112, 112] > 0))

    def test_portrait_image_resize_and_pad(self):
        # 150x300 portrait image (1:2 aspect ratio)
        img = Image.new("RGB", (150, 300), color=(50, 100, 150))
        processed = self.preprocessor.preprocess(img)

        self.assertEqual(processed.size, (224, 224))
        arr = np.array(processed)
        # Left and right should be black padding (0, 0, 0)
        np.testing.assert_array_equal(arr[112, 0], [0, 0, 0])
        np.testing.assert_array_equal(arr[112, 220], [0, 0, 0])
        # Center should not be black
        self.assertTrue(np.any(arr[112, 112] > 0))

    def test_square_image(self):
        img = Image.new("RGB", (300, 300), color=(70, 70, 70))
        processed = self.preprocessor.preprocess(img)
        self.assertEqual(processed.size, (224, 224))

    def test_grayscale_image(self):
        # Grayscale image
        img = Image.new("L", (180, 260), color=100)
        processed = self.preprocessor.preprocess(img)
        self.assertEqual(processed.size, (224, 224))
        self.assertEqual(processed.mode, "L")

    def test_rgba_image(self):
        # RGBA image with alpha channel
        img = Image.new("RGBA", (100, 100), color=(100, 150, 200, 128))
        processed = self.preprocessor.preprocess(img)
        self.assertEqual(processed.size, (224, 224))
        self.assertEqual(processed.mode, "RGBA")

    def test_numpy_array_input_and_output(self):
        # Input numpy array (uint8)
        arr = np.random.randint(0, 256, (120, 180, 3), dtype=np.uint8)
        result = self.preprocessor.preprocess(arr, return_numpy=True)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (224, 224, 3))
        self.assertEqual(result.dtype, np.uint8)

    def test_float_numpy_array_input(self):
        # Float numpy array in [0, 1]
        arr = np.random.rand(100, 100, 3).astype(np.float32)
        result = self.preprocessor(arr, return_numpy=True)
        self.assertEqual(result.shape, (224, 224, 3))
        self.assertEqual(result.dtype, np.uint8)

    def test_file_path_input(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            img = Image.new("RGB", (160, 120), color=(60, 120, 180))
            img.save(tmp_path)

            result = self.preprocessor.preprocess(str(tmp_path))
            self.assertEqual(result.size, (224, 224))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.preprocessor.preprocess("non_existent_image_file.png")

    def test_clahe_enhancement_effect(self):
        # Low contrast image
        low_contrast = np.full((100, 100, 3), 128, dtype=np.uint8)
        # Add slight variation
        low_contrast[25:75, 25:75] = 135
        pil_img = Image.fromarray(low_contrast)

        enhanced = self.preprocessor.apply_clahe(pil_img)
        enhanced_arr = np.array(enhanced)

        # Contrast range should be greater than or equal to original
        orig_range = low_contrast.max() - low_contrast.min()
        enhanced_range = enhanced_arr.max() - enhanced_arr.min()
        self.assertGreaterEqual(enhanced_range, orig_range)


class TestSSIMSimilarity(unittest.TestCase):

    def setUp(self):
        self.preprocessor = ImagePreprocessor()

    def test_identical_images_similarity(self):
        img = Image.new("RGB", (224, 224), color=(100, 150, 200))
        score = compute_ssim_similarity(img, img)
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_different_images_similarity(self):
        np.random.seed(42)
        img1 = np.random.randint(0, 100, (224, 224, 3), dtype=np.uint8)
        img2 = np.random.randint(150, 255, (224, 224, 3), dtype=np.uint8)
        score = compute_ssim_similarity(img1, img2)
        self.assertLess(score, 0.5)

    def test_with_preprocessor_pipeline(self):
        # Test images with different original resolutions
        raw1 = Image.new("RGB", (300, 150), color=(120, 80, 40))
        raw2 = Image.new("RGB", (100, 200), color=(120, 80, 40))

        # Without preprocessor, this would error due to shape mismatch
        with self.assertRaises(ValueError):
            compute_ssim_similarity(raw1, raw2)

        # With preprocessor, both are resized & padded to 224x224
        score = compute_ssim_similarity(raw1, raw2, preprocessor=self.preprocessor)
        self.assertIsInstance(score, float)
        self.assertTrue(-1.0 <= score <= 1.0)

    def test_grayscale_ssim(self):
        img1 = Image.new("L", (224, 224), color=120)
        img2 = Image.new("L", (224, 224), color=120)
        score = compute_ssim_similarity(img1, img2)
        self.assertAlmostEqual(score, 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
