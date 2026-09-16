"""FastAPI Application for Visual Furniture Search (VisionSpace AI).

End-to-End Visual RAG Pipeline:
1. Object Detection: YOLOv8 ONNX with strict COCO classes (chair, couch, potted plant, dining table)
   and class-aware NMS for overlapping furniture.
2. Context-Preserving Cropping: 10-15% margin padding around bounding boxes.
3. Lightweight Visual RAG Search: FastEmbed ONNX embeddings (threads=1) and Qdrant vector database.
4. Clean JSON response with multi-object discovery mapping directly to Next.js and Alpine.js frontends.
"""

from contextlib import asynccontextmanager
import io
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.config import settings
from app.models import (
    DetectedObject,
    ExtractedFeaturesSummary,
    FurnitureDiscoveredItem,
    FurnitureMetadata,
    FurnitureSearchResponse,
    FurnitureSearchResult,
    HealthResponse,
    LensSearchResponse,
    LensVisualMatch,
)
from app.lens_service import lens_service
from app.vector_search import ONNXVisualSearchEngine, search_similar_furniture
from app.vector_service import VectorService
from app.vision_pipeline import VisionPipeline

logger = logging.getLogger("main")

# Global service handles
vision_pipeline: Optional[VisionPipeline] = None
vector_store: Optional[VectorService] = None
vector_search_engine: Optional[ONNXVisualSearchEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models and database connections during application startup."""
    global vision_pipeline, vector_store, vector_search_engine
    print("=== Starting VisionSpaceAI Furniture Search Service ===")

    # 1. Initialize YOLOv8 ONNX Vision Pipeline (CPU Execution Provider, threads=1)
    vision_pipeline = VisionPipeline(device=settings.DEVICE)

    # 2. Initialize Lightweight ONNX Visual Search Engine (Qdrant Cloud / in-memory fallback)
    vector_search_engine = ONNXVisualSearchEngine.get_instance()

    # 3. Initialize Vector Service for backward compatibility and catalog management
    vector_store = VectorService(
        device=settings.DEVICE,
        clip_model=vision_pipeline.clip_model,
        clip_processor=vision_pipeline.clip_processor,
    )
    vector_store.seed_catalog_if_empty(vision_pipeline)

    print("=== Service Ready for Requests ===")
    yield
    print("Shutting down service...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Visual furniture retrieval API using YOLOv8 ONNX multi-object detection, "
        "class-aware NMS, FastEmbed ONNX visual embeddings, and Qdrant vector search."
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
    summary="End-to-End Multi-Object Visual Search Pipeline",
)
async def search_furniture(
    file: UploadFile = File(..., description="Query furniture image file (JPEG, PNG, WEBP)"),
    top_k: int = Form(4, ge=1, le=50, description="Maximum number of nearest furniture items to return per object"),
    category: Optional[str] = Form(None, description="Optional category filter (e.g. Chair, Sofa, Table, Lighting)"),
    min_score: Optional[float] = Form(None, ge=0.0, le=1.0, description="Minimum cosine similarity threshold"),
    include_mobilenet_features: bool = Form(False, description="Whether to also extract MobileNetV2 features"),
):
    """End-to-End Multi-Object Visual RAG Search Pipeline:

    1. Receives uploaded interior / room image.
    2. Runs YOLOv8 ONNX object detection with class-aware NMS to identify all furniture pieces:
       - COCO 56: "chair"
       - COCO 57: "couch"
       - COCO 58: "potted plant"
       - COCO 60: "dining table"
       (Overlapping objects like dining tables in front of couches are both preserved).
    3. Extracts padded crops (10-15% margin) to retain edge and texture context.
    4. Loops through EVERY crop without early returns or slicing, querying Qdrant
       via ONNX FastEmbed visual embeddings.
    5. Returns a structured JSON response mapping directly to Next.js and Alpine.js frontends:
       {
         "items": [
           {"item_name": "couch", "matches": [...]},
           {"item_name": "dining table", "matches": [...]}
         ]
       }
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

    # 2. Multi-Object Detection with YOLOv8 & Padded Crop Extraction
    # Using vision_pipeline with class-aware NMS and 10-15% padding
    if vision_pipeline is not None:
        crops_data = vision_pipeline.detect_and_crop(
            pil_img,
            conf_threshold=0.15,
            iou_threshold=0.45,
            padding_percent=0.12,
        )
    else:
        crops_data = []

    # Fallback: If no objects were detected, treat the full image as a single query
    if not crops_data:
        print("[INFO] [search-furniture] No distinct furniture objects detected. Processing full image as fallback.")
        crops_data = [{
            "crop": pil_img,
            "class_label": "furniture",
            "confidence": 0.85,
            "box": [0, 0, pil_img.width, pil_img.height],
            "padded_box": [0, 0, pil_img.width, pil_img.height],
        }]

    print(f"\n{'='*60}")
    print(f"[STAGE 1] YOLOv8 pipeline extracted {len(crops_data)} crops:")
    for c in crops_data:
        print(f"   * '{c.get('class_label')}' (conf={c.get('confidence')}) box={c.get('box')}")
    print(f"{'='*60}")

    # 3. Process EVERY crop through Vector Search (NO early returns, NO slicing)
    items: List[FurnitureDiscoveredItem] = []
    flattened_results: List[Dict[str, Any]] = []
    detected_objects_summary: List[Dict[str, Any]] = []

    for idx, crop_info in enumerate(crops_data, start=1):
        crop_image = crop_info["crop"]
        class_label = str(crop_info.get("class_label", "furniture"))
        conf = float(crop_info.get("confidence", 0.85))
        box = crop_info.get("box", [0, 0, pil_img.width, pil_img.height])

        # Filter by category if explicitly requested by user in form parameters
        if category and category.strip() and category.lower() != "all":
            if category.lower() not in class_label.lower():
                continue

        # Query Vector Search engine for top-4 similar furniture products
        matches = search_similar_furniture(
            image_crop=crop_image,
            top_k=top_k if top_k else 4,
            category=category if category else None,
            score_threshold=min_score,
        )

        formatted_matches: List[Dict[str, Any]] = []
        for m_idx, m in enumerate(matches, start=1):
            product_name = m.get("product_name", "Furniture Item")
            price_str = str(m.get("price", "Check Website"))
            buy_link = m.get("buy_link", "#")
            store_name = m.get("store_name", "Retailer")
            sim_score = float(m.get("similarity_score", 0.0))
            img_url = m.get("image_url", "")

            match_entry = {
                # Database Payload Metadata (Step 1 schema)
                "product_name": product_name,
                "price": price_str,
                "buy_link": buy_link,
                "store_name": store_name,
                "similarity_score": sim_score,
                "category": m.get("category", class_label.title()),
                "image_url": img_url,
                "description": m.get("description", ""),
                # Front-end UI Compatibility keys (for Next.js & Alpine.js cards)
                "position": m_idx,
                "title": product_name,
                "source": store_name,
                "link": buy_link,
                "thumbnail": img_url,
            }
            formatted_matches.append(match_entry)
            flattened_results.append(match_entry)

        # Build discovered item structure matching both Next.js and Alpine.js
        item_obj = FurnitureDiscoveredItem(
            item_id=idx,
            item_name=class_label,                # Explicitly requested: "couch", "dining table", etc.
            detected_name=class_label.title(),    # UI display title: "Couch", "Dining Table"
            category=class_label.title(),         # Category label
            bbox=box,
            confidence=conf,
            matches=formatted_matches,
        )
        items.append(item_obj)

        detected_objects_summary.append({
            "object_id": idx,
            "label": class_label,
            "category": class_label.title(),
            "confidence": conf,
            "bbox": box,
        })

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    print(f"\n[SUCCESS] [search-furniture] FINAL ITEMS TO FRONTEND: {len(items)}")
    for it in items:
        print(f"   * #{it.item_id} '{it.item_name}' ({it.category}) matches={len(it.matches)}")
    print(f"[TIMING] Search pipeline completed in {elapsed_ms:.2f}ms\n")

    # 4. Return consolidated response
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
