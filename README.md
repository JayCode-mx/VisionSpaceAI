# VisionSpaceAI - Visual Furniture Search Service

A production-ready visual search backend built with **FastAPI**, **PyTorch (Hugging Face Transformers CLIP)**, **Keras (MobileNetV2)**, **OpenCV / Pillow (ImagePreprocessor)**, and **Qdrant Vector Database**.

---

## System Architecture

```
User / Client
      │ (Uploads query image)
      ▼
FastAPI Service (`/api/v1/search-furniture`)
      │
      ▼
ImagePreprocessor (`image_preprocessor.py`)
  ├── OpenCV CLAHE (Contrast-limited adaptive histogram equalization in LAB color space)
  └── Pillow Aspect-Preserving Resize & Letterbox Pad to 224x224
      │
      ├───► PyTorch CLIP (`openai/clip-vit-base-patch32`)
      │       └── Extracts 512-dimensional normalized visual embedding
      │
      └───► Keras MobileNetV2 (ImageNet weights)
              └── Extracts 1280-dimensional deep feature representation
                      │
                      ▼
            Qdrant Vector Database
              ├── Cosine distance similarity index
              ├── Metadata filtering (e.g. category)
              └── Returns Top-K nearest matching furniture items
                      │
                      ▼
               JSON Response (Ranked items, similarity scores, metadata)
```

---

## Features

- **Pillow & OpenCV Preprocessing**:
  - Contrast-limited adaptive histogram equalization (`cv2.createCLAHE`) applied on the Lightness channel ($L$) in LAB space to enhance visual contrast while preserving realistic colors.
  - Aspect-ratio-preserving resize and center-letterbox padding to $224 \times 224$ using Pillow's Lanczos resampling.
- **Dual Deep Vision Models**:
  - **PyTorch Hugging Face Transformers CLIP**: Computes 512-d normalized visual embeddings.
  - **Keras MobileNetV2**: Computes 1280-d deep visual representations for downstream filtering and analysis.
- **Qdrant Vector Search**:
  - In-memory embedded mode (`:memory:`) out of the box with zero external infrastructure required.
  - Seamlessly switches to local persistent directory (`QDRANT_STORAGE_PATH`) or remote cluster (`QDRANT_URL`).
  - Pre-seeded furniture catalog with rich product metadata (name, category, price, dimensions, materials, tags, image URL).
- **FastAPI Endpoint**:
  - `POST /api/v1/search-furniture`: accepts image uploads (`multipart/form-data`), optional `top_k`, `category`, and `min_score` filters.
  - Interactive OpenAPI/Swagger documentation at `/docs`.

---

## Installation & Setup

### 1. Virtual Environment & Dependencies

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate  # On Windows

# Install all dependencies
pip install -r requirements.txt
```

---

## Running the Service

Start the FastAPI application using Uvicorn:

```powershell
.\.venv\Scripts\uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Once running, explore the interactive Swagger documentation at:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## API Endpoints

### 1. Visual Furniture Search
`POST /api/v1/search-furniture`

**Request Parameters (`multipart/form-data`):**
- `file`: Image file (`JPEG`, `PNG`, `WEBP`)
- `top_k` *(optional, default 5)*: Number of nearest items to retrieve ($1 \le \text{top\_k} \le 50$)
- `category` *(optional)*: Filter by category (`Chair`, `Sofa`, `Table`, `Bed`, `Lighting`)
- `min_score` *(optional)*: Minimum cosine similarity score ($0.0$ to $1.0$)
- `include_mobilenet_features` *(optional, default true)*: Whether to compute MobileNetV2 features

**Example cURL:**
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/search-furniture" \
  -F "file=@path/to/chair.jpg" \
  -F "top_k=3" \
  -F "category=Chair"
```

**Example JSON Response:**
```json
{
  "query_id": "62a99057-205b-44c5-8fe8-f4206719f864",
  "total_matches": 1,
  "execution_time_ms": 429.13,
  "features_summary": {
    "clip_embedding_dim": 512,
    "mobilenet_feature_dim": 1280,
    "preprocessor_target_size": [224, 224],
    "clahe_applied": true
  },
  "results": [
    {
      "rank": 1,
      "score": 0.2493,
      "item": {
        "id": "furn-001",
        "name": "Nordic Minimalist Oak Lounge Chair",
        "category": "Chair",
        "price": 349.0,
        "material": "Solid White Oak, Natural Linen",
        "color": "Beige / Light Oak",
        "dimensions": "30W x 32D x 31H in",
        "in_stock": true,
        "tags": ["scandinavian", "minimalist", "accent chair", "wood", "living room"],
        "image_url": "https://images.unsplash.com/photo-1567538096630-e0c55bd6374c",
        "description": "Clean lines and curved solid oak frame paired with organic linen upholstery."
      }
    }
  ]
}
```

### 2. Catalog & Health Endpoints
- `GET /health` or `GET /api/v1/health`: Checks model status and indexed points count in Qdrant.
- `GET /api/v1/furniture`: Returns list of all indexed furniture items in the catalog.

---

## Running Tests and Demonstrations

```powershell
# Run all unit and integration tests (21 tests)
.\.venv\Scripts\python -m unittest discover -s tests

# Run the end-to-end visual search demo script
.\.venv\Scripts\python search_client_demo.py
```
