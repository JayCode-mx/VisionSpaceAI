"""Interactive / client demo for testing the /api/v1/search-furniture endpoint."""

import io
import json
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

from app.main import app


def run_demo():
    print("=== Visual Furniture Search Demo ===\n")

    # 1. Generate sample query image: Scandinavian wooden chair shape
    print("1. Synthesizing a sample furniture query image (Wooden Chair)...")
    img = Image.new("RGB", (320, 240), color=(245, 240, 235))
    draw = ImageDraw.Draw(img)

    # Draw wooden chair structure
    # Seat
    draw.rectangle((90, 130, 230, 150), fill=(185, 122, 87), outline=(139, 69, 19), width=2)
    # Backrest
    draw.rectangle((100, 60, 220, 130), fill=(210, 180, 140), outline=(139, 69, 19), width=2)
    # Legs
    draw.line([(100, 150), (90, 220)], fill=(139, 69, 19), width=6)
    draw.line([(220, 150), (230, 220)], fill=(139, 69, 19), width=6)
    draw.line([(120, 150), (120, 215)], fill=(160, 82, 45), width=4)
    draw.line([(200, 150), (200, 215)], fill=(160, 82, 45), width=4)

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    # 2. Query FastAPI endpoint via TestClient (in-process)
    with TestClient(app) as client:
        print("\n2. Sending POST request to /api/v1/search-furniture (top_k=3)...")
        response = client.post(
            "/api/v1/search-furniture",
            files={"file": ("query_chair.jpg", buf, "image/jpeg")},
            data={"top_k": 3, "include_mobilenet_features": "true"},
        )

        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
            return

        result = response.json()

        print(f"\n3. Search Completed in {result['execution_time_ms']} ms")
        print(f"   Query ID: {result['query_id']}")
        print(f"   Features: CLIP Embedding={result['features_summary']['clip_embedding_dim']}d, "
              f"MobileNet={result['features_summary']['mobilenet_feature_dim']}d, "
              f"Preprocessor Size={result['features_summary']['preprocessor_target_size']}")

        print(f"\n=== Top {len(result['results'])} Nearest Furniture Matches in Qdrant ===")
        for hit in result["results"]:
            item = hit["item"]
            print(f"\n[Rank #{hit['rank']}] Score: {hit['score']:.4f} (Cosine Similarity)")
            print(f"  Name:        {item['name']}")
            print(f"  Category:    {item['category']}")
            print(f"  Price:       ${item['price']:.2f}")
            print(f"  Material:    {item['material']}")
            print(f"  Color:       {item['color']}")
            print(f"  Dimensions:  {item['dimensions']}")
            print(f"  Tags:        {', '.join(item['tags'])}")
            print(f"  Image URL:   {item['image_url']}")


if __name__ == "__main__":
    run_demo()
