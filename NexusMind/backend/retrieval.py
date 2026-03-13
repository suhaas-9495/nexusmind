"""
NexusMind – Retrieval
Performs cosine-similarity search in Qdrant and returns ranked chunks.
"""

import os
from typing import List, Dict

from qdrant_client import QdrantClient
from backend.config import QDRANT_URL, COLLECTION_NAME, TOP_K


_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
    return _client


def retrieve_similar_chunks(
    query_vector: List[float],
    top_k: int | None = None,
    score_threshold: float = 0.0,
    source_filter: str | None = None,
) -> List[Dict]:
    """
    Search Qdrant for the most similar chunks.

    Args:
        query_vector:    Embedded query vector.
        top_k:           Number of results to return.
        score_threshold: Minimum cosine similarity (0-1).
        source_filter:   Optional filename to restrict search to.

    Returns:
        List of payload dicts with an added 'score' key, sorted best-first.
    """
    client = get_client()
    k = top_k or int(os.environ.get("TOP_K", TOP_K))

    query_filter = None
    if source_filter:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        query_filter = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source_filter))]
        )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query          = query_vector,
        limit          = k,
        query_filter   = query_filter,
    ).points

    retrieved = []
    for r in results:
        if r.score < score_threshold:
            continue
        payload = dict(r.payload or {})
        payload["score"] = round(r.score, 4)
        retrieved.append(payload)

    return retrieved
