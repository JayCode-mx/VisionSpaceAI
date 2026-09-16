"""Unit tests for pure NumPy VectorSearch module."""

import numpy as np
from PIL import Image
import pytest

from app.vector_search import (
    VectorSearch,
    vector_db,
    search_similar_furniture,
)


def test_vector_search_engine_initialization():
    """Verify engine initializes with in-memory catalog and zero Qdrant dependency."""
    assert vector_db.get_count() >= 10
    assert len(vector_db.catalog_vectors) == len(vector_db.catalog)


def test_extract_vector_pil():
    """Verify vector extraction from a PIL Image produces a normalized vector."""
    test_img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    vec = vector_db.extract_vector(test_img)

    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-2


def test_extract_vector_numpy():
    """Verify vector extraction from an OpenCV-style BGR NumPy array."""
    cv_array = np.full((120, 160, 3), fill_value=128, dtype=np.uint8)
    vec = vector_db.extract_vector(cv_array)

    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-2


def test_search_similar_furniture_top_4_schema():
    """Verify search_similar_furniture returns exactly top 4 matches with requested payload keys."""
    test_img = Image.new("RGB", (224, 224), color=(80, 50, 30))
    matches = search_similar_furniture(test_img, top_k=4)

    assert len(matches) == 4
    for match in matches:
        assert "product_name" in match
        assert "price" in match
        assert "buy_link" in match
        assert "store_name" in match
        assert "similarity_score" in match

        assert isinstance(match["product_name"], str) and len(match["product_name"]) > 0
        assert isinstance(match["price"], str)
        assert isinstance(match["buy_link"], str)
        assert isinstance(match["store_name"], str)
        assert 0.0 <= match["similarity_score"] <= 1.0


def test_search_category_filtering():
    """Verify category filtering isolates results to the matching COCO label."""
    test_img = Image.new("RGB", (224, 224), color=(80, 50, 30))
    couch_matches = search_similar_furniture(test_img, item_label="couch", top_k=4)
    assert len(couch_matches) == 4
    for m in couch_matches:
        assert m["category"] == "couch"

    table_matches = search_similar_furniture(test_img, item_label="dining table", top_k=4)
    assert len(table_matches) == 4
    for m in table_matches:
        assert m["category"] == "dining table"
