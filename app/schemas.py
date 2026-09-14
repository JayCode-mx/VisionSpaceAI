"""Schemas alias for backwards compatibility with app.models."""

from app.models import (
    FurnitureMetadata,
    FurnitureSearchResult,
    DetectedObject,
    ExtractedFeaturesSummary,
    FurnitureSearchResponse,
    FurnitureDiscoveredItem,
    HealthResponse,
    LensVisualMatch,
    LensSearchResponse,
)

__all__ = [
    "FurnitureMetadata",
    "FurnitureSearchResult",
    "DetectedObject",
    "ExtractedFeaturesSummary",
    "FurnitureSearchResponse",
    "FurnitureDiscoveredItem",
    "HealthResponse",
    "LensVisualMatch",
    "LensSearchResponse",
]
