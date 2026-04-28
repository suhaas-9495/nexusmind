"""
NexusMind – Qdrant Indexing
Creates collection if needed, upserts embeddings + metadata,
and triggers a BM25 index rebuild for hybrid retrieval.
"""

from typing import List, Dict

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    ScrollRequest,
)

from backend.config import QDRANT_URL, COLLECTION_NAME, VECTOR_SIZE


_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
    return _client


def ensure_collection(vector_size: int = VECTOR_SIZE) -> None:
    """Create the Qdrant collection if it doesn't already exist."""
    client = get_client()
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def index_chunks(
    embeddings: List[List[float]],
    metadatas:  List[Dict],
) -> int:
    """
    Upsert (embedding, metadata) pairs into Qdrant AND rebuild BM25 index.
    Returns the number of points upserted.
    """
    client = get_client()
    ensure_collection(vector_size=len(embeddings[0]))

    # Build a stable integer ID from existing count + offset
    existing_count = client.count(collection_name=COLLECTION_NAME).count

    points = [
        PointStruct(
            id      = existing_count + i,
            vector  = vec,
            payload = meta,
        )
        for i, (vec, meta) in enumerate(zip(embeddings, metadatas))
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)

    # ── Rebuild BM25 index over ALL stored chunks ─────────────────────────────
    _rebuild_bm25(client)

    return len(points)


def _rebuild_bm25(client: QdrantClient) -> None:
    """
    Scroll through ALL Qdrant payloads and rebuild the BM25 index.
    This keeps the BM25 and vector store perfectly in sync.
    """
    try:
        from backend.bm25_index import build_bm25_index

        all_chunks: List[Dict] = []
        offset = None

        while True:
            result, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in result:
                if point.payload:
                    all_chunks.append(point.payload)

            if next_offset is None:
                break
            offset = next_offset

        if all_chunks:
            build_bm25_index(all_chunks)

    except Exception as e:
        # BM25 rebuild failure should never crash the indexing pipeline
        import logging
        logging.getLogger(__name__).warning(f"BM25 rebuild failed: {e}")


def delete_collection() -> None:
    """Drop the Qdrant collection and clear the BM25 cache."""
    from pathlib import Path
    client = get_client()
    client.delete_collection(collection_name=COLLECTION_NAME)

    # Clear BM25 cache
    cache = Path("logs/bm25_cache.pkl")
    if cache.exists():
        cache.unlink()
