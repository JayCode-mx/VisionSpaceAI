"""Vector store alias re-exporting VectorService from app.vector_service."""

from app.vector_service import VectorService, VectorStore, DEFAULT_FURNITURE_CATALOG

__all__ = ["VectorService", "VectorStore", "DEFAULT_FURNITURE_CATALOG"]
