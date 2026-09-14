"""Pydantic data models for VisionSpaceAI Furniture Search Service."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FurnitureMetadata(BaseModel):
    id: str = Field(..., description="Unique identifier for the furniture piece")
    name: str = Field(..., description="Product title / name")
    category: str = Field(..., description="Furniture category (e.g. Chair, Sofa, Table, Bed, Lighting, Storage)")
    price: float = Field(..., description="Retail price in USD")
    material: str = Field(..., description="Primary material (e.g. Oak, Velvet, Steel, Leather, Linen)")
    color: str = Field(..., description="Dominant color")
    dimensions: str = Field(..., description="Dimensions (W x D x H)")
    in_stock: bool = Field(True, description="Availability status")
    tags: List[str] = Field(default_factory=list, description="Descriptive tags")
    image_url: Optional[str] = Field(None, description="Reference image URL")
    buy_url: Optional[str] = Field(None, description="Direct product purchasing link")
    source: Optional[str] = Field(None, description="Retailer or e-commerce source (e.g. Amazon, IKEA, Wayfair)")
    description: Optional[str] = Field(None, description="Detailed product description")


class FurnitureSearchResult(BaseModel):
    rank: int = Field(..., description="Rank in similarity search results (1-based)")
    score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
    item: FurnitureMetadata = Field(..., description="Matched furniture item metadata")


class DetectedObject(BaseModel):
    object_id: int = Field(..., description="1-based identifier for detected object")
    label: str = Field(..., description="Detected furniture class (e.g. sofa, chair, dining table)")
    confidence: float = Field(..., description="Detection confidence score (0.0 to 1.0)")
    bbox: List[int] = Field(..., description="Bounding box [x1, y1, x2, y2] in pixels")
    matches: List[FurnitureSearchResult] = Field(default_factory=list, description="Top Qdrant visual matches for this cropped item")


class ExtractedFeaturesSummary(BaseModel):
    clip_embedding_dim: int
    mobilenet_feature_dim: Optional[int] = None
    preprocessor_target_size: tuple[int, int]
    clahe_applied: bool = True


class FurnitureSearchResponse(BaseModel):
    query_id: str = Field(..., description="Unique search request identifier")
    total_objects_detected: int = Field(default=0, description="Total number of objects detected")
    detected_objects: List[DetectedObject] = Field(default_factory=list, description="List of detected objects and their visual matches")
    total_matches: int = Field(default=0, description="Total number of matched items returned (primary/aggregated)")
    results: List[FurnitureSearchResult] = Field(default_factory=list, description="Ranked list of similar furniture for backward compatibility")
    execution_time_ms: float = Field(..., description="End-to-end latency in milliseconds")
    features_summary: Optional[ExtractedFeaturesSummary] = Field(None, description="Visual feature extraction info")


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    device: str
    clip_loaded: bool
    mobilenet_loaded: bool
    qdrant_connected: bool
    indexed_furniture_count: int
