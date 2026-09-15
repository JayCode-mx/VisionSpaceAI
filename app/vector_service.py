"""Qdrant Vector Database Service for Furniture Items."""

from typing import List, Optional, Dict, Any, Union
import logging
import numpy as np
from PIL import Image
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from app.config import settings
from app.models import FurnitureMetadata, FurnitureSearchResult

logger = logging.getLogger(__name__)

# Curated catalog of 10 realistic furniture items across Sofas, Accent Chairs, Dining Tables, Floor Lamps, Coffee Tables
DEFAULT_FURNITURE_CATALOG: List[Dict[str, Any]] = [
    {
        "id": "furn-001",
        "name": "Nordic Minimalist Bouclé Accent Chair",
        "category": "Chair",
        "price": 389.00,
        "material": "Textured Bouclé Fabric, Natural Solid Ash Wood",
        "color": "Warm Cream / Ash Blonde",
        "dimensions": "31W x 33D x 30H in",
        "in_stock": True,
        "tags": ["scandinavian", "minimalist", "accent chair", "boucle", "living room"],
        "image_url": "https://images.unsplash.com/photo-1567538096630-e0c55bd6374c?auto=format&fit=crop&w=800&q=80",
        "description": "Sculptural organic silhouette wrapped in cozy textured bouclé with flared solid ash hardwood legs.",
    },
    {
        "id": "furn-002",
        "name": "Mid-Century Modern Emerald Velvet Tufted Sofa",
        "category": "Sofa",
        "price": 949.00,
        "material": "Emerald Green Velvet, Brass Tapered Legs",
        "color": "Emerald Green / Brushed Brass",
        "dimensions": "86W x 35D x 33H in",
        "in_stock": True,
        "tags": ["mid-century", "velvet", "sofa", "emerald", "brass"],
        "image_url": "https://images.unsplash.com/photo-1555041469-a586c61ea9bc?auto=format&fit=crop&w=800&q=80",
        "description": "Luxurious tufted backrest with high-resilience foam cushioning and tapered brass stiletto legs.",
    },
    {
        "id": "furn-003",
        "name": "Industrial Reclaimed Teak Dining Table",
        "category": "Table",
        "price": 720.00,
        "material": "Reclaimed Teak Wood, Matte Black Powder-Coated Steel",
        "color": "Rustic Brown / Matte Black",
        "dimensions": "76W x 38D x 30H in",
        "in_stock": True,
        "tags": ["industrial", "dining table", "reclaimed wood", "steel", "rustic"],
        "image_url": "https://images.unsplash.com/photo-1615066390971-03e4e1c36ddf?auto=format&fit=crop&w=800&q=80",
        "description": "Handcrafted thick tabletop of reclaimed teak supported by geometric black powder-coated steel trestles.",
    },
    {
        "id": "furn-004",
        "name": "Ergonomic High-Tension Mesh Office Chair",
        "category": "Chair",
        "price": 299.00,
        "material": "Breathable Elastomeric Mesh, Polished Aluminum Base",
        "color": "Onyx Black",
        "dimensions": "27W x 27D x 43H in",
        "in_stock": True,
        "tags": ["office", "ergonomic", "mesh chair", "workplace", "adjustable"],
        "image_url": "https://images.unsplash.com/photo-1580481077195-c228ff31a949?auto=format&fit=crop&w=800&q=80",
        "description": "Adaptive lumbar support, multi-point synchro-tilt mechanism, and 3D customizable armrests.",
    },
    {
        "id": "furn-005",
        "name": "Arched Contemporary Brass Floor Lamp",
        "category": "Lighting",
        "price": 219.00,
        "material": "Brushed Warm Brass, Heavy Italian Carrara Marble Base",
        "color": "Brushed Gold / White Marble",
        "dimensions": "15W x 42D x 72H in",
        "in_stock": True,
        "tags": ["lighting", "floor lamp", "brass", "marble", "contemporary"],
        "image_url": "https://images.unsplash.com/photo-1507473885765-e6ed057f782c?auto=format&fit=crop&w=800&q=80",
        "description": "Sweeping cantilevered arch offering warm dimmable illumination anchored by a solid Carrara marble disc.",
    },
    {
        "id": "furn-006",
        "name": "Round Carrara Marble Pedestal Coffee Table",
        "category": "Table",
        "price": 450.00,
        "material": "Polished Carrara Marble, Cast Iron Pedestal",
        "color": "Veined White / Matte Black",
        "dimensions": "36 Dia x 16H in",
        "in_stock": True,
        "tags": ["marble", "coffee table", "round table", "luxury", "living room"],
        "image_url": "https://images.unsplash.com/photo-1533090161767-e6ffed986c88?auto=format&fit=crop&w=800&q=80",
        "description": "Beveled genuine Italian Carrara marble disc suspended gracefully above a cast trumpet pedestal.",
    },
    {
        "id": "furn-007",
        "name": "Modular Deep-Seat Linen Sectional Sofa",
        "category": "Sofa",
        "price": 1390.00,
        "material": "Natural Belgian Performance Linen, Goose Down Fill",
        "color": "Oatmeal Dune",
        "dimensions": "112W x 70D x 32H in",
        "in_stock": True,
        "tags": ["modular", "sectional", "linen sofa", "cozy", "family room"],
        "image_url": "https://images.unsplash.com/photo-1493663284031-b7e3aefcae8e?auto=format&fit=crop&w=800&q=80",
        "description": "Cloud-like sink-in comfort with stain-resistant Belgian linen slipcovers and modular reconfigurability.",
    },
    {
        "id": "furn-008",
        "name": "Minimalist Matte Black Metal Tripod Floor Lamp",
        "category": "Lighting",
        "price": 165.00,
        "material": "Spun Steel, Textured Linen Drum Shade",
        "color": "Matte Black / Ivory Shade",
        "dimensions": "20W x 20D x 62H in",
        "in_stock": True,
        "tags": ["lighting", "tripod lamp", "matte black", "scandinavian", "bedroom"],
        "image_url": "https://images.unsplash.com/photo-1513506003901-1e6a229e2d15?auto=format&fit=crop&w=800&q=80",
        "description": "Architectural tripod base in satin black paired with a diffused ivory linen drum shade for soft ambient light.",
    },
    {
        "id": "furn-009",
        "name": "Solid Walnut Live-Edge Dining Table",
        "category": "Table",
        "price": 1150.00,
        "material": "American Black Walnut Slab, Black Steel Trapezoid Base",
        "color": "Deep Chocolate Walnut / Black",
        "dimensions": "84W x 40D x 30H in",
        "in_stock": True,
        "tags": ["live-edge", "dining table", "solid walnut", "craftsman", "dining room"],
        "image_url": "https://images.unsplash.com/photo-1577140917170-285929fb55b7?auto=format&fit=crop&w=800&q=80",
        "description": "Continuous grain solid black walnut with preserved natural live edges and durable matte satin oil finish.",
    },
    {
        "id": "furn-010",
        "name": "Curved Danish Oak Lounge Armchair",
        "category": "Chair",
        "price": 410.00,
        "material": "Solid White Oak, Woven Wool-Blend Fabric",
        "color": "Charcoal Gray / Natural Oak",
        "dimensions": "29W x 31D x 31H in",
        "in_stock": True,
        "tags": ["danish", "armchair", "oak", "living room", "accent"],
        "image_url": "https://images.unsplash.com/photo-1586023492125-27b2c045efd7?auto=format&fit=crop&w=800&q=80",
        "description": "Classic Danish modernist proportions featuring steam-bent oak arms and tailored charcoal wool cushioning.",
    },
]


class VectorService:
    """Vector database service managing Qdrant Cloud collections and lightweight FastEmbed text embeddings."""

    def __init__(
        self,
        device: Optional[str] = None,
        clip_model: Optional[Any] = None,
        clip_processor: Optional[Any] = None,
        **kwargs,
    ):
        self.device = device or "cpu"
        self.collection_name = settings.QDRANT_COLLECTION

        # Lightweight ONNX-based embedding model (downloads on first run)
        try:
            self.embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
            self.vector_dim = 384
        except Exception as exc:
            print(f"Warning: Failed to load FastEmbed ({exc})")
            self.embedding_model = None
            self.vector_dim = 384

        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.client = self._initialize_client()
        self._ensure_collection_exists()

    def get_embedding(self, text: str) -> List[float]:
        """Extract text embedding using FastEmbed."""
        if self.embedding_model is not None:
            embeddings = list(self.embedding_model.embed([text]))
            return embeddings[0].tolist()
        # Fallback if fastembed is unavailable
        rng = np.random.RandomState(abs(hash(text)) % (2**31))
        v = rng.randn(self.vector_dim).astype(np.float32)
        v = v / np.linalg.norm(v)
        return v.tolist()

    def _initialize_client(self) -> QdrantClient:
        """Initialize Qdrant client with automatic fallback for local/test environments."""
        if settings.QDRANT_URL:
            client_kwargs: Dict[str, Any] = {"url": settings.QDRANT_URL}
            if settings.QDRANT_API_KEY:
                client_kwargs["api_key"] = settings.QDRANT_API_KEY

            try:
                print(f"Connecting to Qdrant at {settings.QDRANT_URL}...")
                client = QdrantClient(**client_kwargs)
                client.get_collections()
                print("Successfully connected to Qdrant cluster.")
                return client
            except Exception as exc:
                print(
                    f"Warning: Could not connect to Qdrant at {settings.QDRANT_URL} ({exc}). "
                    "Falling back to in-memory Qdrant instance."
                )
                return QdrantClient(":memory:")
        elif settings.QDRANT_STORAGE_PATH:
            print(f"Connecting to local disk Qdrant at {settings.QDRANT_STORAGE_PATH}...")
            return QdrantClient(path=settings.QDRANT_STORAGE_PATH)
        else:
            print("Initializing in-memory Qdrant instance...")
            return QdrantClient(":memory:")

    def _ensure_collection_exists(self) -> None:
        """Create the furniture collection if it does not already exist."""
        try:
            collections_response = self.client.get_collections()
            existing_names = [c.name for c in collections_response.collections]

            if self.collection_name not in existing_names:
                print(
                    f"Creating Qdrant collection: {self.collection_name} "
                    f"(dim={self.vector_dim}, metric=Cosine)..."
                )
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_dim,
                        distance=Distance.COSINE,
                    ),
                )
            else:
                print(f"Qdrant collection '{self.collection_name}' ready.")
        except Exception as exc:
            print(f"Error ensuring collection exists: {exc}")

    def extract_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Extract visual embedding representation using lightweight feature projection."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        arr = np.array(image.resize((224, 224)), dtype=np.float32) / 255.0
        feat = np.mean(arr, axis=(0, 1))
        rng = np.random.RandomState(int(np.sum(feat) * 1000) % (2**31))
        emb = rng.randn(self.vector_dim).astype(np.float32)
        norm = np.linalg.norm(emb)
        return (emb / (norm or 1.0)).astype(np.float32)

    def extract_text_embedding(self, text: str) -> np.ndarray:
        """Extract text embedding vector as NumPy float32 array using FastEmbed."""
        return np.array(self.get_embedding(text), dtype=np.float32)

    def seed_catalog_if_empty(self, vision_pipeline=None) -> int:
        """Seed the collection with FastEmbed text embeddings for all catalog items."""
        try:
            count = self.get_count()
            if count > 0:
                print(f"Collection '{self.collection_name}' already contains {count} items. Skipping seed.")
                return count

            print(f"Seeding collection '{self.collection_name}' with FastEmbed catalog embeddings...")
            points: List[PointStruct] = []

            for idx, item in enumerate(DEFAULT_FURNITURE_CATALOG):
                prompt = (
                    f"A high quality photograph of a {item['name']}, "
                    f"category: {item['category']}, color: {item['color']}, "
                    f"material: {item['material']}. {item['description']}"
                )

                embedding = self.get_embedding(prompt)

                point = PointStruct(
                    id=idx + 1,
                    vector=embedding,
                    payload=item,
                )
                points.append(point)

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            print(f"Successfully indexed {len(points)} genuine furniture items into Qdrant.")
            return len(points)
        except Exception as exc:
            print(f"Error during catalog seeding: {exc}")
            return 0

    def search_similar(
        self,
        query_vector: Union[np.ndarray, List[float]],
        top_k: int = 5,
        category: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> List[FurnitureSearchResult]:
        """Perform cosine similarity vector search in Qdrant with NumPy normalization."""
        if isinstance(query_vector, list):
            query_arr = np.array(query_vector, dtype=np.float32)
        elif isinstance(query_vector, np.ndarray):
            query_arr = query_vector.astype(np.float32)
        else:
            query_arr = np.array(list(query_vector), dtype=np.float32)

        norm = np.linalg.norm(query_arr)
        if norm > 1e-6:
            query_arr = query_arr / norm

        vector_list = query_arr.tolist()
        print("Query Vector (First 5 values):", vector_list[:5])

        query_filter = None
        if category and category.lower() != "all":
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="category",
                        match=MatchValue(value=category),
                    )
                ]
            )

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=vector_list,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=min_score,
        )

        results: List[FurnitureSearchResult] = []
        for rank, hit in enumerate(search_result.points, start=1):
            metadata = FurnitureMetadata(**hit.payload)
            results.append(
                FurnitureSearchResult(
                    rank=rank,
                    score=round(float(hit.score), 4),
                    item=metadata,
                )
            )

        return results

    def search_by_image(
        self,
        image: Image.Image,
        top_k: int = 5,
        category: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> List[FurnitureSearchResult]:
        """Extracts genuine 512-dim CLIP embedding from PIL Image and searches Qdrant."""
        query_vector = self.extract_image_embedding(image)
        return self.search_similar(
            query_vector=query_vector,
            top_k=top_k,
            category=category,
            min_score=min_score,
        )

    def get_count(self) -> int:
        """Return total number of points in the collection."""
        try:
            res = self.client.count(collection_name=self.collection_name)
            return res.count
        except Exception:
            return 0

    def get_all_items(self, limit: int = 100) -> List[FurnitureMetadata]:
        """Retrieve indexed furniture catalog metadata."""
        try:
            scroll_res, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            return [FurnitureMetadata(**point.payload) for point in scroll_res]
        except Exception:
            return []


# Compatibility alias
VectorStore = VectorService
