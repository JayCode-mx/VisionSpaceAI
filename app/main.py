"""FastAPI Application for Visual Furniture Search."""

from contextlib import asynccontextmanager
import io
import time
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.config import settings
from app.schemas import (
    ExtractedFeaturesSummary,
    FurnitureMetadata,
    FurnitureSearchResponse,
    HealthResponse,
)
from app.vector_store import VectorStore
from app.vision_pipeline import VisionPipeline

# Global service handles
vision_pipeline: Optional[VisionPipeline] = None
vector_store: Optional[VectorStore] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models and database connections during application startup."""
    global vision_pipeline, vector_store
    print("=== Starting VisionSpaceAI Furniture Search Service ===")

    # 1. Initialize Vision Pipeline (Preprocessor + CLIP + MobileNetV2)
    vision_pipeline = VisionPipeline(device=settings.DEVICE)

    # 2. Initialize Qdrant Vector Store
    vector_store = VectorStore()

    # 3. Seed Catalog
    vector_store.seed_catalog_if_empty(vision_pipeline)

    print("=== Service Ready for Requests ===")
    yield
    print("Shutting down service...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Visual furniture retrieval API using Pillow/OpenCV CLAHE preprocessing, "
        "PyTorch Hugging Face Transformers CLIP, Keras MobileNetV2, and Qdrant vector search."
    ),
    lifespan=lifespan,
)

# Enable CORS for cross-origin frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    summary="Search furniture using an image",
)
async def search_furniture(
    file: UploadFile = File(..., description="Query furniture image file (JPEG, PNG, WEBP)"),
    top_k: int = Form(5, ge=1, le=50, description="Maximum number of nearest furniture items to return"),
    category: Optional[str] = Form(None, description="Optional category filter (e.g. Chair, Sofa, Table)"),
    min_score: Optional[float] = Form(None, ge=0.0, le=1.0, description="Minimum cosine similarity threshold"),
    include_mobilenet_features: bool = Form(True, description="Whether to also extract Keras MobileNetV2 features"),
):
    """End-to-End Visual Furniture Search:

    1. Receives image upload and reads raw bytes.
    2. Preprocesses through OpenCV CLAHE and Pillow letterbox resize & pad to 224x224.
    3. Extracts 512-dimensional visual embeddings using PyTorch Hugging Face CLIP.
    4. Extracts 1280-dimensional feature representations using Keras MobileNetV2.
    5. Queries Qdrant vector database for top-K nearest matching furniture items.
    6. Returns ranked results with similarity scores and rich product metadata.
    """
    start_time = time.perf_counter()

    if vision_pipeline is None or vector_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vision models or Vector Database are not initialized.",
        )

    # 1. Validate file format
    if not file.content_type or not file.content_type.startswith("image/"):
        # Allow common image file extensions
        valid_extensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
        if not file.filename or not file.filename.lower().endswith(valid_extensions):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type: {file.content_type}. Expected an image file.",
            )

    try:
        contents = await file.read()
        pil_raw_img = Image.open(io.BytesIO(contents))
        pil_raw_img.verify()  # Verify image integrity
        # Re-open after verify() because verify() invalidates the image object
        pil_img = Image.open(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to decode image file: {str(exc)}",
        )

    # 2. Vision Pipeline: Preprocessing (CLAHE + 224x224) + Feature Extraction (CLIP + MobileNetV2)
    try:
        preprocessed_img, clip_emb, mobilenet_feat = vision_pipeline.process_and_extract(
            image_input=pil_img,
            extract_mobilenet=include_mobilenet_features,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vision model inference error: {str(exc)}",
        )

    # 3. Vector Database Search via Qdrant
    try:
        search_results = vector_store.search_similar(
            query_vector=clip_emb,
            top_k=top_k,
            category=category,
            min_score=min_score,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector search failed: {str(exc)}",
        )

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return FurnitureSearchResponse(
        query_id=str(uuid.uuid4()),
        total_matches=len(search_results),
        results=search_results,
        execution_time_ms=round(elapsed_ms, 2),
        features_summary=ExtractedFeaturesSummary(
            clip_embedding_dim=len(clip_emb),
            mobilenet_feature_dim=len(mobilenet_feat) if mobilenet_feat is not None else None,
            preprocessor_target_size=settings.TARGET_IMAGE_SIZE,
            clahe_applied=True,
        ),
    )
