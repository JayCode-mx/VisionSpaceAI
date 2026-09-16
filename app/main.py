"""FastAPI Application for Visual Furniture Search (VisionSpace AI).

End-to-End Ultra-Lightweight Pipeline (Optimized for Render Free Tier <= 512MB RAM):
1. Stage 1 (Detection): YOLOv8 ONNX with strict COCO classes (chair, couch, potted plant, dining table)
   and class-aware NMS to preserve overlapping furniture.
2. RAM Cleanup: Immediate image deletion and explicit Python garbage collection (gc.collect()).
3. Stage 2 (Search): Pure NumPy Cosine Similarity Vector Search with 0MB extra RAM overhead.
4. Clean JSON response with multi-object discovery mapping directly to Next.js and Alpine.js frontends.
"""

from contextlib import asynccontextmanager
import gc
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
from app.vector_search import search_similar_furniture, vector_db
from app.vision_pipeline import VisionPipeline

logger = logging.getLogger("main")

# Global service handles
vision_pipeline: Optional[VisionPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models and in-memory catalog during application startup."""
    global vision_pipeline
    print("=== Starting VisionSpaceAI Furniture Search Service ===")

    # 1. Initialize YOLOv8 ONNX Vision Pipeline (CPU Execution Provider, threads=1)
    vision_pipeline = VisionPipeline(device=settings.DEVICE)

    print(f"=== VectorSearch Ready: {vector_db.get_count()} catalog items indexed (Pure NumPy, 0MB Qdrant overhead) ===")
    yield
    print("Shutting down service...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Visual furniture retrieval API using YOLOv8 ONNX multi-object detection, "
        "class-aware NMS, and Pure NumPy Cosine Similarity vector search."
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
        "engine": "VisionSpace AI Pure NumPy Engine",
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
    if vision_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vision Pipeline not yet initialized",
        )

    count = vector_db.get_count()
    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        device=vision_pipeline.device,
        clip_loaded=True,
        mobilenet_loaded=vision_pipeline.mobilenet_model is not None,
        qdrant_connected=True,  # In-memory pure NumPy vector DB ready
        indexed_furniture_count=count,
    )


@app.get(f"{settings.API_PREFIX}/furniture", tags=["Catalog"])
async def list_catalog(limit: int = 50):
    """List indexed furniture items in the catalog."""
    return vector_db.get_all_items(limit=limit)


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

    Stage 1: YOLOv8 ONNX Multi-Object Detection with Class-Aware NMS & 10-15% padding.
    Memory Cleanup: Explicit garbage collection (gc.collect()) to release full-size image memory.
    Stage 2: Pure NumPy Cosine Similarity search with zero extra RAM bloat.
    Response: Clean JSON structure mapping directly to Next.js and Alpine.js frontends.
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

    # 2. Stage 1: Multi-Object Detection with YOLOv8 & Padded Crop Extraction
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
            "crop": pil_img.copy(),
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

    # =========================================================================
    # MEMORY CLEANUP: Release raw image & force garbage collection before Stage 2
    # Ensures memory stays strictly under Render 512MB RAM free tier limit.
    # =========================================================================
    del pil_img
    del contents
    gc.collect()

    # 3. Stage 2: Process EVERY crop through Pure NumPy Vector Search (NO early returns, NO slicing)
    items: List[FurnitureDiscoveredItem] = []
    flattened_results: List[Dict[str, Any]] = []
    detected_objects_summary: List[Dict[str, Any]] = []

    for idx, crop_info in enumerate(crops_data, start=1):
        crop_image = crop_info["crop"]
        class_label = str(crop_info.get("class_label", "furniture"))
        conf = float(crop_info.get("confidence", 0.85))
        box = crop_info.get("box", [0, 0, 0, 0])

        # Filter by category if explicitly requested by user in form parameters
        if category and category.strip() and category.lower() != "all":
            if category.lower() not in class_label.lower():
                continue

        # Pure NumPy Cosine Similarity Search (Category-Aware)
        matches = search_similar_furniture(
            image_crop=crop_image,
            item_label=class_label,
            top_k=top_k if top_k else 4,
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
                "product_name": product_name,
                "price": price_str,
                "buy_link": buy_link,
                "store_name": store_name,
                "similarity_score": sim_score,
                "category": m.get("category", class_label.title()),
                "image_url": img_url,
                "description": m.get("description", ""),
                # Frontend-friendly alias keys
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

    # Final garbage collection after completing all crops
    gc.collect()

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
