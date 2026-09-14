"""Configuration settings for VisionSpaceAI Furniture Search Service."""

import os
from typing import Optional
from pydantic import BaseModel


class Settings(BaseModel):
    # Service Information
    APP_NAME: str = "VisionSpaceAI - Furniture Visual Search Service"
    APP_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"

    # Vision Models
    CLIP_MODEL_NAME: str = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
    CLIP_EMBEDDING_DIM: int = 512
    MOBILENET_FEATURE_DIM: int = 1280
    DEVICE: str = os.getenv("DEVICE", "cpu")

    # Qdrant Vector DB
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "furniture_items")
    # Set QDRANT_URL to http://localhost:6333 for remote, or leave None for in-memory / local storage
    QDRANT_URL: Optional[str] = os.getenv("QDRANT_URL", None)
    QDRANT_STORAGE_PATH: Optional[str] = os.getenv("QDRANT_STORAGE_PATH", None)

    # Preprocessor defaults
    TARGET_IMAGE_SIZE: tuple[int, int] = (224, 224)
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_TILE_GRID_SIZE: tuple[int, int] = (8, 8)


settings = Settings()
