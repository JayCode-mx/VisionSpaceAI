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


class FurnitureDiscoveredItem(BaseModel):
    item_id: int = Field(..., description="1-based index of the detected item")
    detected_name: str = Field(..., description="Smart name derived from top visual matches or detector")
    category: str = Field(..., description="Furniture category (e.g. Sofa, Table, Chair, Decor)")
    bbox: Optional[List[int]] = Field(None, description="Bounding box [x1, y1, x2, y2]")
    confidence: Optional[float] = Field(None, description="Detection confidence score")
    matches: List[Dict[str, Any]] = Field(default_factory=list, description="Top store buy links and visual matches")


class FurnitureSearchResponse(BaseModel):
    status: str = Field(default="success", description="Status of the search query")
    engine: str = Field(default="VisionSpace Spatial Match v2.0", description="Visual search engine descriptor")
    total_items: int = Field(default=0, description="Total number of furniture items detected")
    execution_time_ms: float = Field(..., description="End-to-end latency in milliseconds")
    items: List[FurnitureDiscoveredItem] = Field(default_factory=list, description="Detected furniture items and matching products")
    total_objects_detected: Optional[int] = Field(None, description="Legacy count of detected objects")
    detected_objects: List[Dict[str, Any]] = Field(default_factory=list, description="Detected objects and bounding boxes")
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Consolidated visual matches")


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    device: str
    clip_loaded: bool
    mobilenet_loaded: bool
    qdrant_connected: bool
    indexed_furniture_count: int


class LensVisualMatch(BaseModel):
    position: int = Field(..., description="Rank / position in visual matches")
    title: str = Field(..., description="Product or webpage title")
    link: str = Field(..., description="Direct link to the matching store or webpage")
    source: Optional[str] = Field(None, description="Retailer or platform name")
    price: Optional[str] = Field(None, description="Formatted price string if available")
    extracted_price: Optional[float] = Field(None, description="Numeric price extracted from listing")
    thumbnail: Optional[str] = Field(None, description="Thumbnail image URL")
    image: Optional[str] = Field(None, description="Full source image URL if available")


class LensSearchResponse(BaseModel):
    engine: str = Field(default="VisionSpace Spatial Match v2.0", description="Visual search engine descriptor")
    total_matches: int
    visual_matches: List[LensVisualMatch]
    execution_time_ms: float

