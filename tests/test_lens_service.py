"""Unit tests for app/lens_service.py ensuring zero mock fallbacks and explicit HTTP 400 errors."""

import io
import unittest
from unittest.mock import MagicMock, patch

from fastapi import HTTPException
from PIL import Image

from app.lens_service import LensService, LensVisualMatch


class TestLensService(unittest.TestCase):
    """Test suite for LensService."""

    def setUp(self):
        # Create a small valid test image in memory
        img = Image.new("RGB", (100, 100), color=(200, 150, 100))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        self.sample_image_bytes = buf.getvalue()

    def test_missing_api_key_raises_http_400(self):
        """Verify that an empty or missing SERPAPI_KEY raises an explicit HTTP 400 exception."""
        service = LensService(api_key="")
        with self.assertRaises(HTTPException) as ctx:
            service.validate_api_key()
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("SERPAPI_KEY is missing", ctx.exception.detail)

    def test_placeholder_api_key_raises_http_400(self):
        """Verify placeholder keys raise explicit HTTP 400."""
        for bad_key in ["YOUR_SERPAPI_KEY", "mock", "placeholder", "test_key"]:
            service = LensService(api_key=bad_key)
            with self.assertRaises(HTTPException) as ctx:
                service.validate_api_key()
            self.assertEqual(ctx.exception.status_code, 400)

    def test_empty_image_bytes_raises_http_400(self):
        """Verify empty image payload raises HTTP 400."""
        service = LensService(api_key="valid_format_key")
        with self.assertRaises(HTTPException) as ctx:
            service.upload_image_to_serpapi(b"")
        self.assertEqual(ctx.exception.status_code, 400)

    def test_serpapi_unauthorized_upload_raises_http_400(self):
        """Verify SerpAPI 401 response raises explicit HTTP 400."""
        service = LensService(api_key="bad_token_12345")

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = '{"error": "Invalid API key"}'
        mock_resp.json.return_value = {"error": "Invalid API key"}

        with patch("requests.post", return_value=mock_resp):
            with self.assertRaises(HTTPException) as ctx:
                service.upload_image_to_serpapi(self.sample_image_bytes)
            self.assertEqual(ctx.exception.status_code, 400)
            self.assertIn("Invalid SERPAPI_KEY", ctx.exception.detail)

    def test_serpapi_unauthorized_search_raises_http_400(self):
        """Verify SerpAPI search 401 raises explicit HTTP 400."""
        service = LensService(api_key="bad_token_12345")

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = '{"error": "Invalid API key"}'
        mock_resp.json.return_value = {"error": "Invalid API key"}

        with patch("requests.get", return_value=mock_resp):
            with self.assertRaises(HTTPException) as ctx:
                service.search_by_image_id("img_123")
            self.assertEqual(ctx.exception.status_code, 400)
            self.assertIn("Invalid SERPAPI_KEY", ctx.exception.detail)

    def test_real_matches_parsing_no_mock_fallback(self):
        """Verify parsing authentic SerpAPI visual matches without mock data."""
        service = LensService(api_key="dummy_key")

        raw_api_data = [
            {
                "position": 1,
                "title": "West Elm Modern Sofa in Forest Green",
                "link": "https://www.westelm.com/products/modern-sofa",
                "source": "West Elm",
                "price": {"extracted_value": 1199.0, "value": "$1,199"},
                "thumbnail": "https://encrypted-tbn2.gstatic.com/images?q=tbn:thumb1",
                "image": "https://www.westelm.com/img1.jpg",
            },
            {
                "position": 2,
                "title": "IKEA Landskrona Tufted Couch",
                "link": "https://www.ikea.com/landskrona",
                "source": "IKEA",
                "price": "$899.00",
                "thumbnail": "https://encrypted-tbn2.gstatic.com/images?q=tbn:thumb2",
            },
        ]

        parsed = service._parse_visual_matches(raw_api_data)
        self.assertEqual(len(parsed), 2)

        item1 = parsed[0]
        self.assertIsInstance(item1, LensVisualMatch)
        self.assertEqual(item1.position, 1)
        self.assertEqual(item1.title, "West Elm Modern Sofa in Forest Green")
        self.assertEqual(item1.link, "https://www.westelm.com/products/modern-sofa")
        self.assertEqual(item1.source, "West Elm")
        self.assertEqual(item1.extracted_price, 1199.0)
        self.assertEqual(item1.price, "$1,199")

        item2 = parsed[1]
        self.assertEqual(item2.position, 2)
        self.assertEqual(item2.title, "IKEA Landskrona Tufted Couch")
        self.assertEqual(item2.source, "IKEA")
        self.assertEqual(item2.extracted_price, 899.0)

    def test_empty_results_returns_empty_list_not_mock(self):
        """Verify that when SerpAPI returns no matches, an empty list is returned instead of mock JSON."""
        service = LensService(api_key="dummy_key")
        parsed = service._parse_visual_matches([])
        self.assertEqual(parsed, [])

    def test_search_multi_crops_and_smart_name_derivation(self):
        """Verify search_multi_crops derives smart labels and returns top 3 matches per crop."""
        service = LensService(api_key="dummy_key")

        crop1 = Image.new("RGB", (100, 100), color=(50, 120, 180))
        crop2 = Image.new("RGB", (100, 100), color=(180, 120, 50))

        crops = [
            {
                "item_id": 1,
                "label": "sofa",
                "category": "Sofa",
                "confidence": 0.91,
                "bbox": [10, 10, 200, 150],
                "cropped_image": crop1,
            },
            {
                "item_id": 2,
                "label": "coffee table",
                "category": "Table",
                "confidence": 0.55,
                "bbox": [50, 100, 150, 180],
                "cropped_image": crop2,
            },
        ]

        def fake_get_visual_matches(crop_bytes):
            if b"\x00" in crop_bytes[:5]:
                pass
            # Return 5 items, search_multi_crops should take top 3
            return [
                {
                    "title": "Article Sven Teal Tufted Velvet 88 Inch Sectional Sofa - West Elm",
                    "source": "West Elm",
                    "price": "$1,499.00",
                    "link": "https://www.westelm.com/sofa",
                    "thumbnail": "https://img.thumb/sofa.jpg",
                },
                {
                    "title": "Modern Teal Velvet Couch",
                    "source": "Wayfair",
                    "price": "$899.00",
                    "link": "https://www.wayfair.com/sofa",
                    "thumbnail": "https://img.thumb/sofa2.jpg",
                },
                {
                    "title": "Mid Century Sectional",
                    "source": "IKEA",
                    "price": "$799.00",
                    "link": "https://www.ikea.com/sofa",
                    "thumbnail": "https://img.thumb/sofa3.jpg",
                },
                {
                    "title": "Extra Couch 4",
                    "source": "Amazon",
                    "price": "$500.00",
                    "link": "https://www.amazon.com/sofa",
                    "thumbnail": "https://img.thumb/sofa4.jpg",
                },
            ]

        with patch.object(service, "get_visual_matches", side_effect=fake_get_visual_matches):
            results = service.search_multi_crops(crops, top_matches_per_crop=3)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["item_id"], 1)
        self.assertIn("Teal Tufted", results[0]["detected_name"])
        self.assertEqual(results[0]["category"], "Sofa")
        self.assertEqual(len(results[0]["matches"]), 3)
        self.assertEqual(results[0]["matches"][0]["source"], "West Elm")

        self.assertEqual(results[1]["item_id"], 2)
        self.assertEqual(len(results[1]["matches"]), 3)

    def test_discard_low_confidence_and_social_media(self):
        """Verify items with confidence < 0.55 are discarded, and Instagram/Reddit results are filtered out."""
        service = LensService(api_key="dummy_key")

        crop_valid = Image.new("RGB", (100, 100), color=(50, 120, 180))
        crop_low_conf = Image.new("RGB", (100, 100), color=(180, 120, 50))

        crops = [
            {
                "item_id": 1,
                "label": "chair",
                "category": "Chair",
                "confidence": 0.85,  # Above 0.55 -> should keep
                "bbox": [10, 10, 100, 100],
                "cropped_image": crop_valid,
            },
            {
                "item_id": 2,
                "label": "table",
                "category": "Table",
                "confidence": 0.33,  # Below 0.55 cutoff -> must discard!
                "bbox": [50, 50, 150, 150],
                "cropped_image": crop_low_conf,
            },
        ]

        def fake_get_visual_matches(crop_bytes):
            return [
                {
                    "title": "Instagram user living room photo",
                    "source": "Instagram",
                    "link": "https://www.instagram.com/p/12345",
                    "thumbnail": "https://instagram.com/th.jpg",
                },
                {
                    "title": "Reddit post about room decor",
                    "source": "Reddit",
                    "link": "https://www.reddit.com/r/CozyPlaces",
                    "thumbnail": "https://reddit.com/th.jpg",
                },
                {
                    "title": "Modern Ergonomic Office Chair",
                    "source": "Herman Miller",
                    "price": "$895.00",
                    "link": "https://www.hermanmiller.com/chair",
                    "thumbnail": "https://hm.com/chair.jpg",
                },
            ]

        with patch.object(service, "get_visual_matches", side_effect=fake_get_visual_matches):
            results = service.search_multi_crops(crops, top_matches_per_crop=3, min_confidence=0.55)

        # Item 2 (confidence 0.33) should be discarded
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["item_id"], 1)
        self.assertEqual(results[0]["category"], "Chair")

        # Instagram and Reddit matches should be filtered out, leaving only Herman Miller
        matches = results[0]["matches"]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["source"], "Herman Miller")
        self.assertNotIn("Instagram", [m["source"] for m in matches])
        self.assertNotIn("Reddit", [m["source"] for m in matches])


if __name__ == "__main__":
    unittest.main()

