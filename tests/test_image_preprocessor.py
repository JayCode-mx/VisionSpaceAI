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


class TestObjectDetection(unittest.TestCase):

    def test_crop_bounding_box(self):
        from image_preprocessor import crop_bounding_box
        img = Image.new("RGB", (300, 200), color=(100, 150, 200))
        crop = crop_bounding_box(img, [50, 40, 150, 120])
        self.assertEqual(crop.size, (100, 80))

        # Test boundary clamping on negative / out-of-bounds coordinates
        crop_clamped = crop_bounding_box(img, [-20, -10, 500, 400])
        self.assertEqual(crop_clamped.size, (300, 200))

    def test_compute_bbox_iou(self):
        from image_preprocessor import compute_bbox_iou
        # Identical boxes -> IoU 1.0
        box = [10, 10, 100, 100]
        self.assertAlmostEqual(compute_bbox_iou(box, box), 1.0, places=4)

        # Non-overlapping boxes -> IoU 0.0
        box2 = [200, 200, 300, 300]
        self.assertEqual(compute_bbox_iou(box, box2), 0.0)

        # 50% overlap
        box3 = [10, 10, 100, 55]
        iou = compute_bbox_iou(box, box3)
        self.assertGreater(iou, 0.4)
        self.assertLess(iou, 0.6)

    def test_interior_classes_and_coco_ids(self):
        from image_preprocessor import INTERIOR_CLASSES, INTERIOR_COCO_IDS
        # Verify all requested COCO interior IDs are present
        required_classes = ["sofa", "chair", "dining table", "tv", "potted plant", "vase", "clock"]
        for rc in required_classes:
            self.assertIn(rc, INTERIOR_CLASSES)

        required_ids = [56, 57, 58, 60, 62, 74, 75]
        for cid in required_ids:
            self.assertIn(cid, INTERIOR_COCO_IDS)

    def test_object_detector_initialization(self):
        from image_preprocessor import ObjectDetector
        detector = ObjectDetector(confidence_threshold=0.20, iou_threshold=0.45)
        self.assertEqual(detector.confidence_threshold, 0.20)
        self.assertEqual(detector.iou_threshold, 0.45)

    def test_sliced_center_crop_fallback_on_single_detection(self):
        from image_preprocessor import ObjectDetector
        detector = ObjectDetector(confidence_threshold=0.20, iou_threshold=0.45)
        # Create a 400x300 room layout image
        img = Image.new("RGB", (400, 300), color=(200, 200, 200))
        detections = detector.detect(img)
        # Verify detect returns a list (runs inference or fallback)
        self.assertIsInstance(detections, list)

    def test_extract_all_furniture_crops_heuristic_fallback(self):
        from image_preprocessor import extract_all_furniture_crops
        from unittest.mock import MagicMock

        # Mock detector detecting only a sofa on the upper side (outside lower-center zone)
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [
            {
                "label": "sofa",
                "category_hint": "Sofa",
                "confidence": 0.88,
                "bbox": [50, 20, 350, 100],
            }
        ]

        img = Image.new("RGB", (400, 300), color=(220, 220, 220))
        crops = extract_all_furniture_crops(img, detector=mock_detector, max_crops=4)

        # Should contain the YOLO sofa crop + heuristic lower-center coffee table crop
        self.assertEqual(len(crops), 2)
        sofa_crop = crops[0]
        self.assertEqual(sofa_crop["label"], "sofa")
        self.assertFalse(sofa_crop["is_heuristic"])

        table_crop = crops[1]
        self.assertEqual(table_crop["label"], "coffee table")
        self.assertEqual(table_crop["category"], "Table")
        self.assertTrue(table_crop["is_heuristic"])
        self.assertEqual(table_crop["bbox"], [100, 135, 300, 255])  # X: 25%-75%, Y: 45%-85%
        self.assertEqual(table_crop["cropped_image"].size, (200, 120))

    def test_extract_all_furniture_crops_with_detected_table(self):
        from image_preprocessor import extract_all_furniture_crops
        from unittest.mock import MagicMock

        # Mock detector detecting a table already in the lower-center zone
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [
            {
                "label": "dining table",
                "category_hint": "Table",
                "confidence": 0.92,
                "bbox": [110, 140, 290, 250],
            }
        ]

        img = Image.new("RGB", (400, 300), color=(220, 220, 220))
        crops = extract_all_furniture_crops(img, detector=mock_detector, max_crops=4)

        # Since table was already in lower-center, no duplicate heuristic coffee table crop should be generated
        self.assertEqual(len(crops), 1)
        self.assertEqual(crops[0]["label"], "dining table")
        self.assertFalse(crops[0]["is_heuristic"])

    def test_target_classes_and_filter_detections(self):
        from image_preprocessor import TARGET_CLASSES, CONF_THRESHOLD, filter_detections
        self.assertEqual(CONF_THRESHOLD, 0.18)
        self.assertIn(60, TARGET_CLASSES)
        self.assertEqual(TARGET_CLASSES[60], "Table")
        self.assertIn(56, TARGET_CLASSES)
        self.assertEqual(TARGET_CLASSES[56], "Chair")
        self.assertIn(57, TARGET_CLASSES)
        self.assertEqual(TARGET_CLASSES[57], "Sofa")
        self.assertIn(58, TARGET_CLASSES)
        self.assertEqual(TARGET_CLASSES[58], "Plant")

        # Test filter_detections
        boxes = [[10, 10, 50, 50], [20, 20, 80, 80], [30, 30, 90, 90], [5, 5, 25, 25]]
        scores = [0.19, 0.12, 0.85, 0.95]
        class_ids = [60, 60, 56, 0]  # class 0 is person (not in TARGET_CLASSES)

        filtered = filter_detections(boxes, scores, class_ids, conf_threshold=0.18)
        # Should keep box 0 (table, 0.19 >= 0.18) and box 2 (chair, 0.85 >= 0.18)
        # Should discard box 1 (table, 0.12 < 0.18) and box 3 (person, not in TARGET_CLASSES)
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]["label"], "Table")
        self.assertEqual(filtered[0]["confidence"], 0.19)
        self.assertEqual(filtered[1]["label"], "Chair")
        self.assertEqual(filtered[1]["confidence"], 0.85)

    def test_crop_with_padding_pil_and_numpy(self):
        from image_preprocessor import crop_with_padding

        # PIL Image test: 200x200 image, box [40, 40, 80, 80] (w=40, h=40)
        # 10% pad = 4px -> [36, 36, 84, 84] (w=48, h=48)
        img = Image.new("RGB", (200, 200), color=(100, 100, 100))
        box = [40, 40, 80, 80]
        cropped_pil = crop_with_padding(img, box, padding_percent=0.10)
        self.assertEqual(cropped_pil.size, (48, 48))

        # Boundary clamping test: box near top-left [2, 2, 20, 20]
        box_edge = [2, 2, 20, 20]
        cropped_edge = crop_with_padding(img, box_edge, padding_percent=0.20)
        self.assertGreater(cropped_edge.size[0], 18)

        # NumPy test: 200x200x3 array
        arr = np.zeros((200, 200, 3), dtype=np.uint8)
        cropped_np = crop_with_padding(arr, box, padding_percent=0.10)
        self.assertEqual(cropped_np.shape, (48, 48, 3))


if __name__ == "__main__":
    unittest.main()

