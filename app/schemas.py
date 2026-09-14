"""Schemas alias for backwards compatibility with app.models."""

from app.models import (
    FurnitureMetadata,
    FurnitureSearchResult,
    DetectedObject,
    ExtractedFeaturesSummary,
    FurnitureSearchResponse,
    HealthResponse,
)

__all__ = [
    "FurnitureMetadata",
    "FurnitureSearchResult",
    "DetectedObject",
    "ExtractedFeaturesSummary",
    "FurnitureSearchResponse",
    "HealthResponse",
]
