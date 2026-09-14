"""FastAPI Application for Visual Furniture Search (VisionSpace AI).

Features Multi-Object Detection using Ultralytics YOLOv8, OpenCV CLAHE preprocessing,
PyTorch Hugging Face Transformers CLIP multimodal embeddings, and Qdrant vector retrieval.
"""

from contextlib import asynccontextmanager
import io
import time
import uuid
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.config import settings
from app.models import (
    DetectedObject,
    ExtractedFeaturesSummary,
    FurnitureMetadata,
    FurnitureSearchResult,
    FurnitureSearchResponse,
    FurnitureDiscoveredItem,
    HealthResponse,
    LensSearchResponse,
    LensVisualMatch,
)
from app.lens_service import lens_service
from app.object_detector import ObjectDetector
from app.image_preprocessor import extract_all_furniture_crops
from app.vector_service import VectorService
from app.vision_pipeline import VisionPipeline

# Global service handles
vision_pipeline: Optional[VisionPipeline] = None
vector_store: Optional[VectorService] = None
object_detector: Optional[ObjectDetector] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models and database connections during application startup."""
    global vision_pipeline, vector_store, object_detector
    print("=== Starting VisionSpaceAI Furniture Search Service ===")

    # 1. Initialize Vision Pipeline (Preprocessor + CLIP + MobileNetV2)
    vision_pipeline = VisionPipeline(device=settings.DEVICE)

    # 2. Initialize Qdrant Vector Store (sharing preloaded CLIP model)
    vector_store = VectorService(
        device=settings.DEVICE,
        clip_model=vision_pipeline.clip_model,
        clip_processor=vision_pipeline.clip_processor,
    )

    # 3. Initialize YOLOv8 Object Detector with lower threshold for low-profile furniture
    object_detector = ObjectDetector(model_name="yolov8n.pt", confidence_threshold=0.20, iou_threshold=0.45)

    # 4. Seed Catalog if collection is empty
    vector_store.seed_catalog_if_empty(vision_pipeline)

    print("=== Service Ready for Requests ===")
    yield
    print("Shutting down service...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Visual furniture retrieval API using Ultralytics YOLOv8 multi-object detection, "
        "OpenCV CLAHE preprocessing, PyTorch CLIP embeddings, and Qdrant vector search."
    ),
    lifespan=lifespan,
)

# Parse allowed origins for CORS middleware (from CORS_ORIGINS or ALLOWED_ORIGINS)
raw_origins = getattr(settings, "CORS_ORIGINS", None) or getattr(settings, "ALLOWED_ORIGINS", "*")
if isinstance(raw_origins, str):
    parsed_origins = [orig.strip() for orig in raw_origins.split(",") if orig.strip()]
    origins = parsed_origins if parsed_origins else ["*"]
elif isinstance(raw_origins, (list, tuple)):
    origins = list(raw_origins)
else:
    origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root():
    """Lightweight health check endpoint for Render health checks and general discovery."""
    return {
        "status": "online",
        "engine": "VisionSpace AI Neural Engine",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs_url": "/docs",
        "search_endpoint": f"{settings.API_PREFIX}/search-furniture",
        "health_endpoint": f"{settings.API_PREFIX}/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
@app.get(f"{settings.API_PREFIX}/health", response_model=HealthResponse, tags=["Diagnostics"])
async def health_check():
    """Service health and model readiness check."""
    if vision_pipeline is None or vector_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models or Vector Database not yet initialized",
        )

    count = vector_store.get_count()
    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        device=vision_pipeline.device,
        clip_loaded=vision_pipeline.clip_model is not None,
        mobilenet_loaded=vision_pipeline.mobilenet_model is not None,
        qdrant_connected=True,
        indexed_furniture_count=count,
    )


@app.get(f"{settings.API_PREFIX}/furniture", response_model=list[FurnitureMetadata], tags=["Catalog"])
async def list_catalog(limit: int = 50):
    """List indexed furniture items in the catalog."""
    if vector_store is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Vector store unavailable")
    return vector_store.get_all_items(limit=limit)


@app.post(
    f"{settings.API_PREFIX}/search-furniture",
    response_model=FurnitureSearchResponse,
    status_code=status.HTTP_200_OK,
    tags=["Visual Search"],
    summary="Multi-object visual search for furniture using an image",
)
async def search_furniture(
    file: UploadFile = File(..., description="Query furniture image file (JPEG, PNG, WEBP)"),
    top_k: int = Form(5, ge=1, le=50, description="Maximum number of nearest furniture items to return per object"),
    category: Optional[str] = Form(None, description="Optional category filter (e.g. Chair, Sofa, Table, Lighting)"),
    min_score: Optional[float] = Form(None, ge=0.0, le=1.0, description="Minimum cosine similarity threshold"),
    include_mobilenet_features: bool = Form(True, description="Whether to also extract MobileNetV2 features"),
):
    """End-to-End Multi-Object Visual Search Pipeline:

    1. Reads uploaded image bytes and decodes PIL Image.
    2. Runs Ultralytics YOLOv8 object detection to identify interior & furniture objects (sofa, chair, table, etc.).
    3. For each detected object bounding box:
       - Crops the sub-region.
       - Preprocesses crop via OpenCV CLAHE and aspect-ratio letterboxing to 224x224.
       - Computes 512-dim normalized PyTorch CLIP visual embedding.
       - Queries Qdrant vector database for top matching furniture items.
    4. Fallback: If no objects are detected, processes the full image as a single query.
    5. Returns ranked matches per detected object with bounding boxes and confidence scores.
    """
    start_time = time.perf_counter()

    # 1. Validate file format
    if not file.content_type or not file.content_type.startswith("image/"):
        valid_extensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
        if not file.filename or not file.filename.lower().endswith(valid_extensions):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type: {file.content_type}. Expected an image file.",
            )

    try:
        contents = await file.read()
        if not contents:
            raise ValueError("Uploaded image file is empty.")
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read/decode uploaded image: {str(exc)}",
        )

    # 2. Hybrid Region Extraction (YOLOv8 + Heuristic Region Sampling)
    crops = extract_all_furniture_crops(pil_img, detector=object_detector, max_crops=4)

    # 3. Unified Visual Search across all crops
    top_matches = top_k if top_k else 3
    discovered_items_raw = lens_service.search_multi_crops(crops, top_matches_per_crop=top_matches)

    # Filter by category if requested
    if category and category.strip():
        cat_lower = category.strip().lower()
        filtered = [it for it in discovered_items_raw if cat_lower in it.get("category", "").lower()]
        if filtered:
            discovered_items_raw = filtered

    # 4. Consolidate into structured response
    items: List[FurnitureDiscoveredItem] = []
    flattened_results: List[Dict[str, Any]] = []
    detected_objects_summary: List[Dict[str, Any]] = []

    for it in discovered_items_raw:
        item_obj = FurnitureDiscoveredItem(
            item_id=it["item_id"],
            detected_name=it["detected_name"],
            category=it["category"],
            bbox=it.get("bbox"),
            confidence=it.get("confidence"),
            matches=it.get("matches", []),
        )
        items.append(item_obj)
        flattened_results.extend(it.get("matches", []))
        if it.get("bbox"):
            detected_objects_summary.append({
                "object_id": it["item_id"],
                "label": it["detected_name"],
                "category": it["category"],
                "confidence": it.get("confidence", 0.85),
                "bbox": it.get("bbox"),
            })

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return FurnitureSearchResponse(
        status="success",
        engine="VisionSpace Spatial Match v2.0",
        total_items=len(items),
        total_objects_detected=len(items),
        execution_time_ms=round(elapsed_ms, 2),
        items=items,
        detected_objects=detected_objects_summary,
        results=flattened_results,
    )


@app.post(
    f"{settings.API_PREFIX}/search-lens",
    response_model=LensSearchResponse,
    status_code=status.HTTP_200_OK,
    tags=["Visual Search"],
    summary="Search real visual matches using spatial discovery engine",
)
async def search_lens(
    file: UploadFile = File(..., description="Query image file to search for visual matches"),
):
    """High-precision visual search endpoint.

    Executes live visual discovery across verified merchant catalogs.
    Raises explicit HTTP 400 if search credentials are unconfigured or invalid.
    """
    start_time = time.perf_counter()

    if not file.content_type or not file.content_type.startswith("image/"):
        valid_extensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
        if not file.filename or not file.filename.lower().endswith(valid_extensions):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type: {file.content_type}. Expected an image file.",
            )

    try:
        contents = await file.read()
        if not contents:
            raise ValueError("Uploaded image file is empty.")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read/decode uploaded image: {str(exc)}",
        )

    # Execute live visual search
    visual_matches = lens_service.search_lens(
        image_input=contents,
        filename=file.filename or "query.jpg",
        content_type=file.content_type or "image/jpeg",
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return LensSearchResponse(
        engine="VisionSpace Spatial Match v2.0",
        total_matches=len(visual_matches),
        visual_matches=visual_matches,
        execution_time_ms=round(elapsed_ms, 2),
    )

