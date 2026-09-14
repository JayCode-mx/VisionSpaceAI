"""Integration tests for FastAPI Furniture Visual Search Service."""

import io
import unittest
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

    def test_search_furniture_success(self):
        img_bytes = self._create_sample_image()
        response = self.client.post(
            "/api/v1/search-furniture",
            files={"file": ("query_furniture.png", img_bytes, "image/png")},
            data={"top_k": 3, "include_mobilenet_features": "true"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("query_id", data)
        self.assertEqual(data["total_matches"], 3)
        self.assertEqual(len(data["results"]), 3)

        # Check top result
        top_match = data["results"][0]
        self.assertEqual(top_match["rank"], 1)
        self.assertIsInstance(top_match["score"], float)
        self.assertIn("name", top_match["item"])
        self.assertIn("category", top_match["item"])
        self.assertIn("price", top_match["item"])

        # Check features summary
        features = data["features_summary"]
        self.assertEqual(features["clip_embedding_dim"], 512)
        self.assertEqual(features["mobilenet_feature_dim"], 1280)
        self.assertEqual(features["preprocessor_target_size"], [224, 224])
        self.assertTrue(features["clahe_applied"])

    def test_search_furniture_with_category_filter(self):
        img_bytes = self._create_sample_image()
        response = self.client.post(
            "/api/v1/search-furniture",
            files={"file": ("query_chair.png", img_bytes, "image/png")},
            data={"top_k": 5, "category": "Chair"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertGreater(len(data["results"]), 0)
        for match in data["results"]:
            self.assertEqual(match["item"]["category"], "Chair")

    def test_search_furniture_invalid_file(self):
        fake_file = io.BytesIO(b"This is not a real image file.")
        response = self.client.post(
            "/api/v1/search-furniture",
            files={"file": ("test.txt", fake_file, "text/plain")},
        )
        self.assertEqual(response.status_code, 400)

    def test_multi_object_response_structure(self):
        img_bytes = self._create_sample_image(size=(400, 300))
        response = self.client.post(
            "/api/v1/search-furniture",
            files={"file": ("room_scene.png", img_bytes, "image/png")},
            data={"top_k": 2},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("total_objects_detected", data)
        self.assertIn("detected_objects", data)
        self.assertGreater(data["total_objects_detected"], 0)
        self.assertGreater(len(data["detected_objects"]), 0)

        first_obj = data["detected_objects"][0]
        self.assertIn("object_id", first_obj)
        self.assertIn("label", first_obj)
        self.assertIn("confidence", first_obj)
        self.assertIn("bbox", first_obj)
        self.assertEqual(len(first_obj["bbox"]), 4)
        self.assertIn("matches", first_obj)
        self.assertGreater(len(first_obj["matches"]), 0)


if __name__ == "__main__":
    unittest.main()
