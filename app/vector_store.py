"""Qdrant Vector Database Store for Furniture Items."""

from typing import List, Optional
import uuid
import numpy as np
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
from app.schemas import FurnitureMetadata, FurnitureSearchResult


# Catalog of diverse furniture items to seed the vector database
DEFAULT_FURNITURE_CATALOG = [
    {
        "id": "furn-001",
        "name": "Nordic Minimalist Oak Lounge Chair",
        "category": "Chair",
        "price": 349.00,
        "material": "Solid White Oak, Natural Linen",
        "color": "Beige / Light Oak",
        "dimensions": "30W x 32D x 31H in",
        "in_stock": True,
        "tags": ["scandinavian", "minimalist", "accent chair", "wood", "living room"],
        "image_url": "https://images.unsplash.com/photo-1567538096630-e0c55bd6374c",
        "description": "Clean lines and curved solid oak frame paired with organic linen upholstery.",
    },
    {
        "id": "furn-002",
        "name": "Mid-Century Modern Velvet Tufted Sofa",
        "category": "Sofa",
        "price": 899.00,
        "material": "Emerald Green Velvet, Brass Tapered Legs",
        "color": "Emerald Green",
        "dimensions": "84W x 35D x 33H in",
        "in_stock": True,
        "tags": ["mid-century", "velvet", "sofa", "emerald", "brass"],
        "image_url": "https://images.unsplash.com/photo-1555041469-a586c61ea9bc",
        "description": "Luxurious tufted backrest with plush high-density foam cushions and angled brass legs.",
    },
    {
        "id": "furn-003",
        "name": "Industrial Reclaimed Wood Dining Table",
        "category": "Table",
        "price": 680.00,
        "material": "Reclaimed Teak, Matte Black Powder-Coated Steel",
        "color": "Rustic Brown / Black",
        "dimensions": "72W x 36D x 30H in",
        "in_stock": True,
        "tags": ["industrial", "dining table", "reclaimed wood", "steel", "rustic"],
        "image_url": "https://images.unsplash.com/photo-1615066390971-03e4e1c36ddf",
        "description": "Heavy-duty steel trestle base supporting a handcrafted rustic reclaimed teak tabletop.",
    },
    {
        "id": "furn-004",
        "name": "Ergonomic Mesh Swivel Office Task Chair",
        "category": "Chair",
        "price": 289.00,
        "material": "Breathable High-Tension Mesh, Aluminum Base",
        "color": "Black",
        "dimensions": "26W x 26D x 42H in",
        "in_stock": True,
        "tags": ["office", "ergonomic", "mesh chair", "workplace", "adjustable"],
        "image_url": "https://images.unsplash.com/photo-1580481077195-c228ff31a949",
        "description": "Dynamic lumbar support, multi-angle synchro-tilt, and 3D adjustable armrests.",
    },
    {
        "id": "furn-005",
        "name": "Minimalist Japanese Platform Bed Frame",
        "category": "Bed",
        "price": 750.00,
        "material": "Solid Walnut Hardwood",
        "color": "Dark Walnut",
        "dimensions": "82L x 64W x 14H in (Queen)",
        "in_stock": True,
        "tags": ["zen", "platform bed", "walnut", "bedroom", "low profile"],
        "image_url": "https://images.unsplash.com/photo-1505693416388-ac5ce068fe85",
        "description": "Low-profile floating platform bed crafted from sustainably sourced American black walnut.",
    },
    {
        "id": "furn-006",
        "name": "Arched Brass Floor Reading Lamp",
        "category": "Lighting",
        "price": 195.00,
        "material": "Brushed Brass, Heavy White Marble Base",
        "color": "Gold / White",
        "dimensions": "14W x 38D x 68H in",
        "in_stock": True,
        "tags": ["lighting", "floor lamp", "brass", "marble", "contemporary"],
        "image_url": "https://images.unsplash.com/photo-1507473885765-e6ed057f782c",
        "description": "Sweeping cantilevered arched arm with warm rotary-dimmable LED illumination.",
    },
    {
        "id": "furn-007",
        "name": "Modular Sectional Linen Sofa in Cloud Gray",
        "category": "Sofa",
        "price": 1250.00,
        "material": "Performance Linen, Feather Down Blend",
        "color": "Light Gray",
        "dimensions": "108W x 68D x 32H in",
        "in_stock": True,
        "tags": ["modular", "sectional", "cloud sofa", "cozy", "family room"],
        "image_url": "https://images.unsplash.com/photo-1493663284031-b7e3aefcae8e",
        "description": "Deep seats with ultra-soft down-blend cushions and stain-resistant performance weave.",
    },
    {
        "id": "furn-008",
        "name": "Round Carrara Marble Coffee Table",
        "category": "Table",
        "price": 420.00,
        "material": "Authentic Carrara Marble, Cast Iron Pedestal",
        "color": "White Veined / Black",
        "dimensions": "36 Dia x 16H in",
        "in_stock": True,
        "tags": ["marble", "coffee table", "round table", "luxury", "living room"],
        "image_url": "https://images.unsplash.com/photo-1533090161767-e6ffed986c88",
        "description": "Polished Italian Carrara marble disc atop a sculptured trumpet base.",
    },
]


class VectorStore:
    """Manages connection, collection lifecycle, indexing, and vector similarity search with Qdrant."""

    def __init__(self):
        if settings.QDRANT_URL:
            print(f"Connecting to remote Qdrant at {settings.QDRANT_URL}...")
            self.client = QdrantClient(url=settings.QDRANT_URL)
        elif settings.QDRANT_STORAGE_PATH:
            print(f"Connecting to local disk Qdrant at {settings.QDRANT_STORAGE_PATH}...")
            self.client = QdrantClient(path=settings.QDRANT_STORAGE_PATH)
        else:
            print("Initializing in-memory Qdrant instance...")
            self.client = QdrantClient(":memory:")

        self.collection_name = settings.QDRANT_COLLECTION
        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        """Create the furniture collection if it does not already exist."""
        collections_response = self.client.get_collections()
        existing_names = [c.name for c in collections_response.collections]

        if self.collection_name not in existing_names:
            print(f"Creating Qdrant collection: {self.collection_name} (dim={settings.CLIP_EMBEDDING_DIM}, metric=Cosine)...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.CLIP_EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
        else:
            print(f"Qdrant collection '{self.collection_name}' already exists.")

    def seed_catalog_if_empty(self, vision_pipeline) -> int:
        """Seed the collection with default furniture items if empty.

        Uses CLIP text embeddings derived from item name, category, and visual description
        to align seamlessly with visual embeddings in shared CLIP latent space.
        """
        count = self.get_count()
        if count > 0:
            print(f"Collection already has {count} items. Skipping initial seeding.")
            return count

        print("Seeding Qdrant collection with furniture catalog...")
        points: List[PointStruct] = []

        for idx, item in enumerate(DEFAULT_FURNITURE_CATALOG):
            # Formulate a descriptive prompt that anchors CLIP visual search
            prompt = (
                f"A high quality studio photograph of a {item['name']}, "
                f"category: {item['category']}, color: {item['color']}, "
                f"material: {item['material']}. {item['description']}"
            )
            # Generate normalized CLIP embedding
            embedding = vision_pipeline.extract_clip_text_embedding(prompt)

            point = PointStruct(
                id=idx + 1,
                vector=embedding.tolist(),
                payload=item,
            )
            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        print(f"Successfully indexed {len(points)} furniture items into Qdrant.")
        return len(points)

    def search_similar(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        category: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> List[FurnitureSearchResult]:
        """Perform cosine similarity vector search in Qdrant.

        Args:
            query_vector: 512-d normalized query embedding from CLIP.
            top_k: Number of nearest items to retrieve.
            category: Optional category filter (e.g. 'Chair', 'Sofa', 'Table').
            min_score: Minimum cosine similarity threshold.

        Returns:
            List[FurnitureSearchResult]: Ranked furniture items.
        """
        query_filter = None
        if category:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="category",
                        match=MatchValue(value=category),
                    )
                ]
            )

        # Convert numpy vector to python list
        vector_list = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector

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

    def get_count(self) -> int:
        """Return total number of points in the collection."""
        res = self.client.count(collection_name=self.collection_name)
        return res.count

    def get_all_items(self, limit: int = 100) -> List[FurnitureMetadata]:
        """Retrieve indexed furniture catalog metadata."""
        scroll_res, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [FurnitureMetadata(**point.payload) for point in scroll_res]
