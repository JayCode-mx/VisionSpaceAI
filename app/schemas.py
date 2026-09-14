"""Pydantic schemas for VisionSpaceAI Furniture Search Service."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FurnitureMetadata(BaseModel):
    id: str = Field(..., description="Unique identifier for the furniture piece")
    name: str = Field(..., description="Product title / name")
    category: str = Field(..., description="Furniture category (e.g. Chair, Sofa, Table, Bed, Storage)")
    price: float = Field(..., description="Retail price in USD")
    material: str = Field(..., description="Primary material (e.g. Oak, Velvet, Steel, Leather)")
    color: str = Field(..., description="Dominant color")
    dimensions: str = Field(..., description="Dimensions (W x D x H)")
    in_stock: bool = Field(True, description="Availability status")
    tags: List[str] = Field(default_factory=list, description="Descriptive tags")
    image_url: Optional[str] = Field(None, description="Reference image URL")
    description: Optional[str] = Field(None, description="Detailed product description")


class FurnitureSearchResult(BaseModel):
    rank: int = Field(..., description="Rank in similarity search results (1-based)")
    score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
    item: FurnitureMetadata = Field(..., description="Matched furniture item metadata")


class ExtractedFeaturesSummary(BaseModel):
    clip_embedding_dim: int
    mobilenet_feature_dim: Optional[int] = None
    preprocessor_target_size: tuple[int, int]
    clahe_applied: bool = True


class FurnitureSearchResponse(BaseModel):
    query_id: str = Field(..., description="Unique search request identifier")
    total_matches: int = Field(..., description="Number of matched items returned")
    results: List[FurnitureSearchResult] = Field(..., description="Ranked list of similar furniture")
    execution_time_ms: float = Field(..., description="End-to-end latency in milliseconds")
    features_summary: ExtractedFeaturesSummary = Field(..., description="Visual feature extraction info")


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    device: str
    clip_loaded: bool
    mobilenet_loaded: bool
    qdrant_connected: bool
    indexed_furniture_count: int
