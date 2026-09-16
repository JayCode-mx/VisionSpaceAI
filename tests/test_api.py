"""Integration tests for FastAPI Furniture Visual Search Service."""

import io
import unittest
from unittest import mock
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

from app.main import app


class TestFurnitureSearchAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Using context manager enters lifespan and initializes models + Qdrant
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)

    def _create_sample_image(self, size=(300, 200), color=(100, 150, 200)) -> io.BytesIO:
        """Helper to create an in-memory PNG image."""
        buf = io.BytesIO()
        img = Image.new("RGB", size, color=color)
        draw = ImageDraw.Draw(img)
        draw.rectangle((50, 40, 250, 160), fill=(180, 120, 80))
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertEqual(data["engine"], "VisionSpace AI Neural Engine")
        self.assertIn("service", data)
        self.assertIn("docs_url", data)

    def test_health_check(self):
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["clip_loaded"])
        self.assertTrue(data["mobilenet_loaded"])
        self.assertTrue(data["qdrant_connected"])
        self.assertGreater(data["indexed_furniture_count"], 0)

    def test_list_furniture_catalog(self):
        response = self.client.get("/api/v1/furniture")
        self.assertEqual(response.status_code, 200)
        items = response.json()
        self.assertIsInstance(items, list)
        self.assertGreater(len(items), 0)
        first_item = items[0]
        self.assertIn("id", first_item)
        self.assertIn("name", first_item)
        self.assertIn("category", first_item)
        self.assertIn("price", first_item)

    @mock.patch("app.main.search_furniture_with_lens")
    def test_search_furniture_success(self, mock_lens):
        mock_lens.return_value = [
            {
                "title": "Modern Velvet Armchair",
                "source": "Wayfair",
                "price": "$299.00",
                "link": "https://www.wayfair.com/furniture/pdp/chair-123.html",
                "thumbnail": "https://serpapi.com/th?q=chair",
            },
            {
                "title": "Mid-Century Lounge Chair",
                "source": "IKEA",
                "price": "$179.99",
                "link": "https://www.ikea.com/item-456.html",
                "thumbnail": "https://serpapi.com/th?q=ikea-chair",
            },
            {
                "title": "Nordic Accent Chair",
                "source": "Amazon",
                "price": "$145.50",
                "link": "https://www.amazon.com/dp/B0123456",
                "thumbnail": "https://serpapi.com/th?q=amazon-chair",
            },
        ]

        img_bytes = self._create_sample_image()
        response = self.client.post(
            "/api/v1/search-furniture",
            files={"file": ("query_furniture.png", img_bytes, "image/png")},
            data={"top_k": 3},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["status"], "success")
        self.assertEqual(data["engine"], "VisionSpace Spatial Match v2.0")
        self.assertGreaterEqual(data["total_objects_detected"], 1)
        self.assertIn("detected_objects", data)
        self.assertIsInstance(data["execution_time_ms"], float)
        self.assertIsInstance(data["results"], list)
        self.assertEqual(len(data["results"]), 3)

        # Check top result
        top_match = data["results"][0]
        self.assertEqual(top_match["title"], "Modern Velvet Armchair")
        self.assertEqual(top_match["source"], "Wayfair")
        self.assertEqual(top_match["price"], "$299.00")
        self.assertEqual(top_match["link"], "https://www.wayfair.com/furniture/pdp/chair-123.html")
        self.assertEqual(top_match["thumbnail"], "https://serpapi.com/th?q=chair")

    @mock.patch("app.main.search_furniture_with_lens")
    def test_search_furniture_with_category_filter(self, mock_lens):
        mock_lens.return_value = [
            {
                "title": "Dining Chair Set of 2",
                "source": "Target",
                "price": "$120.00",
                "link": "https://www.target.com/p/chair",
                "thumbnail": "https://serpapi.com/th?q=target-chair",
            }
        ]
        img_bytes = self._create_sample_image()
        response = self.client.post(
            "/api/v1/search-furniture",
            files={"file": ("query_chair.png", img_bytes, "image/png")},
            data={"top_k": 5, "category": "Chair"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["status"], "success")
        self.assertGreater(len(data["results"]), 0)
        self.assertEqual(data["results"][0]["title"], "Dining Chair Set of 2")

    def test_search_furniture_invalid_file(self):
        fake_file = io.BytesIO(b"This is not a real image file.")
        response = self.client.post(
            "/api/v1/search-furniture",
            files={"file": ("test.txt", fake_file, "text/plain")},
        )
        self.assertEqual(response.status_code, 400)

    @mock.patch("app.main.search_furniture_with_lens")
    def test_search_furniture_lens_results(self, mock_lens):
        mock_lens.return_value = [
            {
                "title": "Minimalist Coffee Table",
                "source": "Article",
                "price": "$349.00",
                "link": "https://www.article.com/table",
                "thumbnail": "https://serpapi.com/th?q=table",
            }
        ]
        img_bytes = self._create_sample_image(size=(400, 300))
        response = self.client.post(
            "/api/v1/search-furniture",
            files={"file": ("room_scene.png", img_bytes, "image/png")},
            data={"top_k": 2},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["status"], "success")
        self.assertIn("total_objects_detected", data)
        self.assertIn("execution_time_ms", data)
        self.assertIn("results", data)
        self.assertEqual(len(data["results"]), 1)
        self.assertEqual(data["results"][0]["title"], "Minimalist Coffee Table")

    @mock.patch("app.main.search_furniture_with_lens")
    @mock.patch("app.main.vision_pipeline.detect_and_crop")
    def test_search_furniture_consolidated_items_response(self, mock_extract, mock_lens):
        from PIL import Image
        mock_extract.return_value = [
            {
                "crop": Image.new("RGB", (100, 100)),
                "class_label": "couch",
                "confidence": 0.89,
                "box": [20, 30, 200, 150],
            },
            {
                "crop": Image.new("RGB", (100, 100)),
                "class_label": "dining table",
                "confidence": 0.75,
                "box": [60, 120, 180, 220],
            },
        ]
        def fake_lens(image_crop, top_k=4, category=None):
            if category == "couch":
                return [{
                    "title": "Article Sven Teal Tufted Sofa",
                    "source": "West Elm",
                    "price": "$1,499.00",
                    "link": "https://www.westelm.com/sofa",
                    "thumbnail": "https://img.thumb/sofa.jpg",
                }]
            return [{
                "title": "Tiered Coffee Table",
                "source": "Wayfair",
                "price": "$299.00",
                "link": "https://www.wayfair.com/table",
                "thumbnail": "https://img.thumb/table.jpg",
            }]
        mock_lens.side_effect = fake_lens

        img_bytes = self._create_sample_image(size=(400, 300))
        response = self.client.post(
            "/api/v1/search-furniture",
            files={"file": ("living_room.png", img_bytes, "image/png")},
            data={"top_k": 3},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["status"], "success")
        self.assertEqual(data["total_items"], 2)
        self.assertIn("execution_time_ms", data)
        self.assertIn("items", data)
        self.assertEqual(len(data["items"]), 2)

        # Check item 1
        item1 = data["items"][0]
        self.assertEqual(item1["item_id"], 1)
        self.assertEqual(item1["detected_name"], "Couch")
        self.assertEqual(item1["item_name"], "couch")
        self.assertEqual(len(item1["matches"]), 1)
        self.assertEqual(item1["matches"][0]["source"], "West Elm")

        # Check item 2
        item2 = data["items"][1]
        self.assertEqual(item2["item_id"], 2)
        self.assertEqual(item2["detected_name"], "Dining Table")
        self.assertEqual(item2["item_name"], "dining table")
        self.assertEqual(len(item2["matches"]), 1)
        self.assertEqual(item2["matches"][0]["source"], "Wayfair")


if __name__ == "__main__":
    unittest.main()

