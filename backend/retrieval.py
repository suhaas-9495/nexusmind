"""
NexusMind v2 — Retrieval
Cosine-similarity search in Qdrant with optional user_id access control.
"""

import os
from typing import List, Dict, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from backend.config import QDRANT_URL, COLLECTION_NAME, TOP_K

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
    return _client


def retrieve_similar_chunks(
    query_vector:    List[float],
    top_k:           Optional[int] = None,
    score_threshold: float = 0.0,
    source_filter:   Optional[str] = None,
    user_id:         Optional[str] = None,    # 🔐 document-level access control
) -> List[Dict]:
    """
    Search Qdrant for the most similar chunks.
    When user_id is provided, only returns chunks uploaded by that user.
    """
    client = get_client()
    k = top_k or int(os.environ.get("TOP_K", TOP_K))

    # Build filter: combine source_filter AND user_id if both provided
    must_conditions = []
    if source_filter:
        must_conditions.append(FieldCondition(key="source", match=MatchValue(value=source_filter)))
    if user_id:
        must_conditions.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))

    query_filter = Filter(must=must_conditions) if must_conditions else None

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=k,
        query_filter=query_filter,
    ).points

    retrieved = []
    for r in results:
        if r.score < score_threshold:
            continue
        payload = dict(r.payload or {})
        payload["score"] = round(r.score, 4)
        retrieved.append(payload)

    return retrieved
