# VisionSpace AI — Multimodal Visual Furniture Search Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch CPU](https://img.shields.io/badge/PyTorch-CPU%20Optimized-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-CLAHE-5C3EE8.svg?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Qdrant Cloud](https://img.shields.io/badge/Qdrant-Cloud%20Vector%20DB-DC2626.svg?style=flat&logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Render Backend](https://img.shields.io/badge/Render-Free%20Tier%20Ready-46E3B7.svg?style=flat&logo=render&logoColor=black)](https://render.com/)
[![GitHub Pages](https://img.shields.io/badge/Frontend-GitHub%20Pages-222222.svg?style=flat&logo=githubpages&logoColor=white)](https://pages.github.com/)

**VisionSpace AI** is a production-ready, full-stack visual search platform for designer furniture. It normalizes variable lighting using OpenCV Contrast Limited Adaptive Histogram Equalization (CLAHE), extracts 512-dimensional multimodal latent embeddings via PyTorch CLIP, and executes high-speed cosine similarity retrieval over Qdrant Cloud.

Architected specifically for **100% free-tier cloud deployment**:
- **Backend Service**: FastAPI & PyTorch on **Render** (CPU-only optimized, <512MB RAM footprint).
- **Vector Database**: **Qdrant Cloud** (Managed free-tier cluster, 1GB vector storage).
- **Frontend Client**: Responsive dark-theme SPA on **GitHub Pages** (Tailwind CSS + Alpine.js).

---

## 🌐 Live Deployments & Demos

| Component | Target Platform | Live URL | Status |
|---|---|---|---|
| **Frontend SPA** | GitHub Pages | `https://jaycode-mx.github.io/VisionSpaceAI/frontend/` | [![Status](https://img.shields.io/badge/Status-Live-emerald.svg)](https://jaycode-mx.github.io/VisionSpaceAI/frontend/) |
| **Backend API** | Render Web Service | `https://visionspace-ai.onrender.com` | [![Status](https://img.shields.io/badge/Status-Live-emerald.svg)](https://visionspace-ai.onrender.com) |
| **Interactive Docs** | FastAPI Swagger UI | `https://visionspace-ai.onrender.com/docs` | [![Swagger](https://img.shields.io/badge/Docs-Swagger-009688.svg)](https://visionspace-ai.onrender.com/docs) |
| **Vector Database** | Qdrant Cloud | `https://<your-cluster-id>.qdrant.tech:6333` | [![Qdrant](https://img.shields.io/badge/Cluster-Active-DC2626.svg)](https://cloud.qdrant.io/) |

---

## 🏗️ System Architecture

```
                                  [ USER QUERY ]
                  (Dropzone Image Upload via GitHub Pages SPA)
                                        │
                                        ▼
                     ┌─────────────────────────────────────┐
                     │   FastAPI Engine (Render Cloud)     │
                     │    POST /api/v1/search-furniture    │
                     └──────────────────┬──────────────────┘
                                        │
                                        ▼
                     ┌─────────────────────────────────────┐
                     │     app/image_preprocessor.py       │
                     │  1. cv2.cvtColor(RGB -> LAB)        │
                     │  2. CLAHE on L-channel (clip=2.0)   │
                     │  3. cv2.cvtColor(LAB -> RGB)        │
                     │  4. Pillow Lanczos pad to 224x224   │
                     └──────────────────┬──────────────────┘
                                        │
                                        ▼
                     ┌─────────────────────────────────────┐
                     │       PyTorch CLIP Latent Space     │
                     │   Model: openai/clip-vit-base-p32   │
                     │   Extract: 512-dim visual vector    │
                     │   Normalize: L2 Unit Vector (norm=1)│
                     └──────────────────┬──────────────────┘
                                        │
                                        ▼
                     ┌─────────────────────────────────────┐
                     │    Qdrant Cloud Vector Database     │
                     │   Collection: 'furniture' (dim=512) │
                     │   Distance Metric: COSINE           │
                     │   Payload Filter: category (opt.)   │
                     └──────────────────┬──────────────────┘
                                        │
                                        ▼
                     ┌─────────────────────────────────────┐
                     │     Ranked Response & Telemetry     │
                     │   - Similarity percentage (e.g. 92%)│
                     │   - Unsplash product imagery        │
                     │   - Materials, pricing, dimensions  │
                     │   - Execution latency (sub-500ms)   │
                     └─────────────────────────────────────┘
```

---

## ⚡ Key Technical Capabilities

1. **Adaptive Histogram Equalization (OpenCV CLAHE)**
   Raw user photographs often suffer from uneven ambient lighting, harsh shadows, or underexposure. The preprocessor isolates the luminance ($L$) channel in the LAB color space, calculates localized adaptive histogram transformations with a clip limit of 2.0 on an $8\times 8$ grid, and re-merges chrominance channels. This guarantees optimal feature fidelity without altering authentic color palette values.

2. **Aspect-Preserving Letterbox Resizing (Pillow)**
   Rather than performing aggressive isotropic stretching that deforms furniture proportions (e.g. turning tall lamps into squat shapes), images are centered and padded to $224 \times 224$ using high-quality Lanczos resampling with zero distortion.

3. **Multimodal CLIP Embeddings (512-D Latent Space)**
   Leverages OpenAI's Vision Transformer (`openai/clip-vit-base-patch32`). Both text descriptions and product imagery project onto a unified 512-dimensional hypersphere, enabling cross-modal zero-shot similarity matching.

4. **Sub-500ms Cosine Distance Retrieval (Qdrant)**
   Leverages HNSW indexing and payload filtering directly inside Qdrant Cloud. Category filtering (`Sofa`, `Chair`, `Table`, `Lighting`) executes concurrently with vector distance calculations in under 20ms.

5. **Resource-Throttled CPU Optimization for Free-Tier Deployment**
   All deep learning dependencies are locked to CPU-only execution (`--extra-index-url https://download.pytorch.org/whl/cpu`), eliminating GPU driver overhead and keeping memory consumption under Render's 512MB threshold.

---

## 📁 Repository Structure

```
VisionSpaceAI/
├── app/
│   ├── __init__.py                # Package declaration
│   ├── config.py                  # Pydantic Settings & environment variables
│   ├── main.py                    # FastAPI application, CORS & visual search routes
│   ├── image_preprocessor.py      # OpenCV CLAHE & Pillow letterbox padding pipeline
│   ├── vector_service.py          # Qdrant client lifecycle, fallback & vector search
│   └── models.py                  # Pydantic schemas (Metadata, SearchResult, Health)
├── frontend/
│   └── index.html                 # Single-file responsive dark-theme SPA (Tailwind + Alpine)
├── scripts/
│   └── seed_qdrant.py             # Automated Qdrant collection creator & catalog seeder
├── tests/
│   ├── test_api.py                # FastAPI endpoint integration tests
│   └── test_image_preprocessor.py # CLAHE & letterboxing unit test suite
├── .gitignore                     # Production ignore rules
├── pytest.ini                     # Pytest discovery & warning suppression
├── requirements.txt               # CPU-optimized production dependency manifest
├── render.yaml                    # 1-click Render Infrastructure-as-Code blueprint
└── README.md                      # Engineering documentation
```

---

## 🚀 Local Development Setup

### 1. Prerequisites
- Python 3.11+ (recommended 3.11 or 3.12)
- Git

### 2. Clone & Virtual Environment

```bash
git clone https://github.com/JayCode-mx/VisionSpaceAI.git
cd VisionSpaceAI

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install CPU-optimized dependencies
pip install -r requirements.txt
```

### 3. Environment Variables (Optional for Local)

Create a `.env` file in the project root:

```env
# Optional: Set remote Qdrant credentials, or leave default for in-memory fallback
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=furniture
ALLOWED_ORIGINS=*
```

> **Note**: If Qdrant is not running locally on port 6333, `VectorService` automatically falls back to an embedded `:memory:` instance for seamless local testing without Docker!

### 4. Seed the Database

Run the automated seeder to initialize the `furniture` collection and index 10 curated catalog items:

```bash
python scripts/seed_qdrant.py
```

### 5. Launch the Backend API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- API Root: [http://localhost:8000](http://localhost:8000)
- Interactive Swagger Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

### 6. Run the Frontend Locally

Simply open `frontend/index.html` in any web browser, or serve it with Python:

```bash
python -m http.server 3000 --directory frontend
```

Navigate to [http://localhost:3000](http://localhost:3000). The frontend automatically connects to `http://localhost:8000`.

---

## 🧪 Automated Testing

Execute the complete automated test suite using `pytest`:

```bash
python -m pytest
```

Output:
```
============================= test session starts =============================
platform win32 -- Python 3.14.6, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\JOY\Desktop\VisionSpaceAI
configfile: pytest.ini
testpaths: tests
plugins: anyio-4.15.1
collected 21 items

tests\test_api.py ......                                                 [ 28%]
tests\test_image_preprocessor.py ...............                         [100%]

============================= 21 passed in 27.88s =============================
```

---

## ☁️ Cloud Deployment Guide (Free-Tier)

### Step 1: Provision Free Qdrant Cloud Cluster

1. Sign up at [cloud.qdrant.io](https://cloud.qdrant.io/).
2. Create a free 1GB cluster.
3. Copy your **Cluster URL** (e.g., `https://xyz-abc.eu-central.aws.cloud.qdrant.io:6333`) and generate an **API Key**.

### Step 2: Seed the Qdrant Cloud Collection

From your local machine or terminal, run the seeding script with your Qdrant Cloud credentials:

```bash
export QDRANT_URL="https://your-cluster-id.cloud.qdrant.io:6333"
export QDRANT_API_KEY="your-qdrant-api-key"
python scripts/seed_qdrant.py
```

### Step 3: Deploy Backend on Render (1-Click Blueprint)

1. Push your repository to GitHub:
   ```bash
   git push -u origin main
   ```
2. Log into [dashboard.render.com](https://dashboard.render.com/).
3. Click **New +** -> **Blueprint**.
4. Connect the `VisionSpaceAI` repository. Render will automatically read `render.yaml`.
5. Under Environment Variables, input:
   - `QDRANT_URL`: `https://your-cluster-id.cloud.qdrant.io:6333`
   - `QDRANT_API_KEY`: `your-qdrant-api-key`
   - `ALLOWED_ORIGINS`: `*` (or your GitHub Pages domain)
6. Click **Apply**. Render will install CPU dependencies, start Uvicorn, and provide your public URL (e.g. `https://visionspace-ai.onrender.com`).

### Step 4: Deploy Frontend to GitHub Pages

1. In your GitHub repository, navigate to **Settings** -> **Pages**.
2. Under **Build and deployment**:
   - Source: `Deploy from a branch`
   - Branch: `main`
   - Folder: `/ (root)` or `/frontend`
3. Click **Save**.
4. Open the hosted page (e.g. `https://jaycode-mx.github.io/VisionSpaceAI/frontend/`).
5. Click the gear icon in the header and set your Backend API URL to your live Render endpoint. The status beacon will turn **Emerald ("Online")**.

---

## 🔌 API Reference

### 1. Visual Search
`POST /api/v1/search-furniture`

**Form Parameters:**
| Parameter | Type | Required | Description |
|---|---|---|---|
| `file` | Binary File | Yes | Image file (JPEG, PNG, WEBP) |
| `top_k` | Integer | No (Default: 5) | Maximum nearest neighbors to retrieve (1-50) |
| `category` | String | No | Category filter (`Sofa`, `Chair`, `Table`, `Lighting`) |
| `min_score` | Float | No | Minimum cosine similarity threshold (0.0 - 1.0) |

**Sample cURL Request:**
```bash
curl -X POST "https://visionspace-ai.onrender.com/api/v1/search-furniture" \
  -F "file=@living_room_chair.jpg" \
  -F "top_k=3" \
  -F "category=Chair"
```

**Sample Response (`200 OK`):**
```json
{
  "query_id": "8d3ea5f6-c567-4da1-96f3-4411d7395eb9",
  "total_matches": 3,
  "execution_time_ms": 218.45,
  "features_summary": {
    "clip_embedding_dim": 512,
    "mobilenet_feature_dim": 1280,
    "preprocessor_target_size": [224, 224],
    "clahe_applied": true
  },
  "results": [
    {
      "rank": 1,
      "score": 0.8924,
      "item": {
        "id": "furn-001",
        "name": "Nordic Minimalist Bouclé Accent Chair",
        "category": "Chair",
        "price": 389.0,
        "material": "Textured Bouclé Fabric, Natural Solid Ash Wood",
        "color": "Warm Cream / Ash Blonde",
        "dimensions": "31W x 33D x 30H in",
        "in_stock": true,
        "tags": ["scandinavian", "minimalist", "accent chair", "boucle"],
        "image_url": "https://images.unsplash.com/photo-1567538096630-e0c55bd6374c",
        "description": "Sculptural organic silhouette wrapped in cozy textured bouclé."
      }
    }
  ]
}
```

### 2. Service Health Check
`GET /api/v1/health`

Returns engine readiness, model status, and total indexed catalog items.

---

## ⚖️ License

Distributed under the Apache 2.0 License. See `LICENSE` for details.
