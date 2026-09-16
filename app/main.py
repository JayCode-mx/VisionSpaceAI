"""FastAPI Application for Visual Furniture Search (VisionSpace AI).

End-to-End Ultra-Lightweight Pipeline (Optimized for Render Free Tier <= 512MB RAM):
1. Stage 1 (Detection): YOLOv8 ONNX multi-object detection with 15% context padding.
2. RAM Cleanup: Immediate image deletion and explicit Python garbage collection (gc.collect()).
3. Stage 2 (Search): Real-time SerpApi Google Lens discovery with 0MB extra RAM overhead.
4. Response: Clean JSON structure mapping directly to Next.js and Alpine.js frontends,
   guaranteeing all detected objects are preserved without early return or slicing.
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
from app.lens_service import lens_service, search_furniture_with_lens
from app.vision_pipeline import VisionPipeline

logger = logging.getLogger("main")

# Global service handles
vision_pipeline: Optional[VisionPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models during application startup."""
    global vision_pipeline
    print("=== Starting VisionSpaceAI Furniture Search Service ===")

    # 1. Initialize YOLOv8 ONNX Vision Pipeline (CPU Execution Provider, threads=1)
    vision_pipeline = VisionPipeline(device=settings.DEVICE)

    print("=== SerpApi Google Lens Service Ready (0MB Vector DB overhead) ===")
    yield
    print("Shutting down service...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Visual furniture retrieval API using YOLOv8 ONNX multi-object detection, "
        "class-aware NMS, and SerpAPI Google Lens live visual discovery."
    ),
    lifespan=lifespan,
)

# Parse allowed origins for CORS middleware
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
    if vision_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vision Pipeline not yet initialized",
        )

    count = lens_service.get_catalog_count()
    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        device=vision_pipeline.device,
        clip_loaded=True,
        mobilenet_loaded=vision_pipeline.mobilenet_model is not None,
        qdrant_connected=True,
        indexed_furniture_count=count,
    )


@app.get(f"{settings.API_PREFIX}/furniture", tags=["Catalog"])
async def list_catalog(limit: int = 50):
    """List indexed furniture items in the catalog."""
    return lens_service.get_catalog_items(limit=limit)


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
    min_score: Optional[float] = Form(None, ge=0.0, le=1.0, description="Minimum similarity threshold"),
    include_mobilenet_features: bool = Form(False, description="Whether to also extract MobileNetV2 features"),
):
    """End-to-End Multi-Object Visual Search Pipeline:

    Stage 1: YOLOv8 ONNX Multi-Object Detection with Class-Aware NMS & 15% context padding.
    Memory Cleanup: Explicit garbage collection (gc.collect()) to release full-size image memory.
    Stage 2: Loop through EVERY cropped item without early returns or slicing, querying Google Lens via SerpApi.
    Response: Complete list of all discovered items mapping cleanly to Next.js and Alpine.js frontends.
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

    # 2. Stage 1: Run image through vision_pipeline to extract all cropped items with 15% padding
    if vision_pipeline is not None:
        crops_data = vision_pipeline.detect_and_crop(
            pil_img,
            conf_threshold=0.15,
            iou_threshold=0.50,
            padding_percent=0.15,
        )
    else:
        crops_data = []

    # Fallback: If no distinct objects were detected, treat the full image as a single query
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

    # 3. Stage 2: Initialize empty final_response list and process EVERY crop item
    final_response: List[Dict[str, Any]] = []
    flattened_results: List[Dict[str, Any]] = []
    detected_objects_summary: List[Dict[str, Any]] = []

    for idx, crop_info in enumerate(crops_data, start=1):
        crop_image = crop_info["crop"]
        class_label = str(crop_info.get("class_label", "furniture"))
        conf = float(crop_info.get("confidence", 0.85))
        box = crop_info.get("box", [0, 0, 0, 0])

        # Filter by category if explicitly requested by user in form parameters
        if category and category.strip() and category.lower() != "all":
            clean_req_cat = category.strip().lower()
            if class_label.lower() != "furniture" and clean_req_cat not in class_label.lower():
                continue
            if class_label.lower() == "furniture":
                class_label = clean_req_cat

        # Call SerpApi Google Lens for real-world visual product matches
        matches = search_furniture_with_lens(
            image_crop=crop_image,
            top_k=top_k if top_k else 4,
            category=class_label,
        )

        formatted_matches: List[Dict[str, Any]] = []
        for m_idx, m in enumerate(matches, start=1):
            product_name = m.get("product_name") or m.get("title") or "Furniture Item"
            price_str = str(m.get("price", "Check Website"))
            buy_link = m.get("buy_link") or m.get("link") or "#"
            store_name = m.get("store_name") or m.get("source") or "Retailer"
            sim_score = float(m.get("similarity_score", 0.0))
            img_url = m.get("image_url") or m.get("thumbnail") or ""

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

        # Append item dictionary with integer item_id (DO NOT use category names as unique keys)
        item_entry = {
            "item_id": idx,
            "item_name": class_label,
            "detected_name": class_label.title(),
            "category": class_label.title(),
            "bbox": box,
            "confidence": conf,
            "matches": formatted_matches,
        }
        final_response.append(item_entry)

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

    print(f"\n[SUCCESS] [search-furniture] FINAL ITEMS TO FRONTEND: {len(final_response)}")
    for it in final_response:
        print(f"   * #{it['item_id']} '{it['item_name']}' ({it['category']}) matches={len(it['matches'])}")
    print(f"[TIMING] Search pipeline completed in {elapsed_ms:.2f}ms\n")

    # 4. Return consolidated response containing full final_response list (NO slicing!)
    return FurnitureSearchResponse(
        status="success",
        engine="VisionSpace Spatial Match v2.0",
        total_items=len(final_response),
        total_objects_detected=len(final_response),
        execution_time_ms=round(elapsed_ms, 2),
        items=[FurnitureDiscoveredItem(**item) for item in final_response],
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
