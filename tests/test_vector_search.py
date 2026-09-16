"""Unit tests for the lightweight ONNX Visual RAG vector_search module."""

import numpy as np
from PIL import Image
import pytest

from app.vector_search import (
    ONNXVisualSearchEngine,
    search_similar_furniture,
    VECTOR_DIMENSION,
)


def test_vector_search_engine_initialization():
    """Verify engine initializes with correct vector dimension and collection."""
    engine = ONNXVisualSearchEngine.get_instance()
    assert engine.vector_dim == VECTOR_DIMENSION
    assert engine.vector_dim == 512
    assert engine.client is not None


def test_extract_image_embedding_pil():
    """Verify embedding generation from a PIL Image produces a 512-dim unit vector."""
    engine = ONNXVisualSearchEngine.get_instance()
    test_img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    emb = engine.extract_image_embedding(test_img)

    assert isinstance(emb, np.ndarray)
    assert emb.shape == (512,)
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-2


def test_extract_image_embedding_numpy():
    """Verify embedding generation from an OpenCV-style BGR NumPy array."""
    engine = ONNXVisualSearchEngine.get_instance()
    cv_array = np.full((120, 160, 3), fill_value=128, dtype=np.uint8)
    emb = engine.extract_image_embedding(cv_array)

    assert isinstance(emb, np.ndarray)
    assert emb.shape == (512,)
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-2


def test_search_similar_furniture_top_4_schema():
    """Verify search_similar_furniture returns exactly top 4 matches with requested payload keys."""
    test_img = Image.new("RGB", (224, 224), color=(80, 50, 30))
    matches = search_similar_furniture(test_img, top_k=4)

    assert len(matches) == 4
    for match in matches:
        # Check required schema keys
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
