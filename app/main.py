"""FastAPI Application for Visual Furniture Search (VisionSpace AI).

Features Multi-Object Detection using Ultralytics YOLOv8, OpenCV CLAHE preprocessing,
PyTorch Hugging Face Transformers CLIP multimodal embeddings, and Qdrant vector retrieval.
"""

from contextlib import asynccontextmanager
import io
import time
import uuid
from typing import Optional, List

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
    HealthResponse,
)
from app.object_detector import ObjectDetector
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

    # 3. Initialize YOLOv8 Object Detector
    object_detector = ObjectDetector(model_name="yolov8n.pt", confidence_threshold=0.25)

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

# Parse allowed origins for CORS middleware
raw_origins = settings.ALLOWED_ORIGINS
if isinstance(raw_origins, str):
    parsed_origins = [orig.strip() for orig in raw_origins.split(",") if orig.strip()]
    origins = parsed_origins if parsed_origins else ["*"]
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
    """Root landing endpoint."""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs_url": "/docs",
        "search_endpoint": f"{settings.API_PREFIX}/search-furniture",
        "health_endpoint": f"{settings.API_PREFIX}/health",
        "status": "online",
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

    if vision_pipeline is None or vector_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vision models or Vector Database are not initialized.",
        )

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

    # 2. Multi-Object Detection using YOLOv8
    detected_raw: List[dict] = []
    if object_detector is not None:
        try:
            detected_raw = object_detector.detect(pil_img)
        except Exception as exc:
            print(f"Warning: Object detector failed ({exc}), falling back to full image.")
            detected_raw = []

    detected_objects: List[DetectedObject] = []
    primary_clip_emb = None
    mobilenet_feat = None

    # 3. Process Detections or Fallback to Full Image
    if detected_raw and len(detected_raw) > 0:
        print(f"Detected {len(detected_raw)} furniture/interior objects in image.")
        for det in detected_raw:
            crop_img = det["cropped_image"]
            # Preprocess cropped item and extract CLIP visual embedding
            _, crop_clip_emb, _ = vision_pipeline.process_and_extract(
                image_input=crop_img,
                extract_mobilenet=False,
            )

            if primary_clip_emb is None:
                primary_clip_emb = crop_clip_emb

            # Determine category filter: user explicit category takes precedence, else hint from YOLO
            eff_category = category if category else det.get("category_hint")

            matches = vector_store.search_similar(
                query_vector=crop_clip_emb,
                top_k=top_k,
                category=eff_category,
                min_score=min_score,
            )

            # Ensure buy_url is populated with product buy links / image_url fallback
            for match in matches:
                if not match.item.buy_url and match.item.image_url:
                    match.item.buy_url = match.item.image_url

            detected_objects.append(
                DetectedObject(
                    object_id=det["object_id"],
                    label=det["label"],
                    confidence=det["confidence"],
                    bbox=det["bbox"],
                    matches=matches,
                )
            )
    else:
        # Fallback: Process full image
        print("No individual objects detected. Operating in full-image fallback mode.")
        preprocessed_img, clip_emb, mobilenet_feat = vision_pipeline.process_and_extract(
            image_input=contents,
            extract_mobilenet=include_mobilenet_features,
        )
        primary_clip_emb = clip_emb
        query_vector = clip_emb
        print("Query Vector (First 5 values):", query_vector[:5])

        matches = vector_store.search_similar(
            query_vector=clip_emb,
            top_k=top_k,
            category=category,
            min_score=min_score,
        )

        for match in matches:
            if not match.item.buy_url and match.item.image_url:
                match.item.buy_url = match.item.image_url

        w, h = pil_img.size
        detected_objects.append(
            DetectedObject(
                object_id=1,
                label="full image",
                confidence=1.0,
                bbox=[0, 0, w, h],
                matches=matches,
            )
        )

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # Aggregate primary matches for backwards compatibility
    primary_matches = detected_objects[0].matches if detected_objects else []

    return FurnitureSearchResponse(
        query_id=str(uuid.uuid4()),
        total_objects_detected=len(detected_objects),
        detected_objects=detected_objects,
        total_matches=len(primary_matches),
        results=primary_matches,
        execution_time_ms=round(elapsed_ms, 2),
        features_summary=ExtractedFeaturesSummary(
            clip_embedding_dim=len(primary_clip_emb) if primary_clip_emb is not None else 512,
            mobilenet_feature_dim=len(mobilenet_feat) if mobilenet_feat is not None else None,
            preprocessor_target_size=settings.TARGET_IMAGE_SIZE,
            clahe_applied=True,
        ),
    )
