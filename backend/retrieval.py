import os
from typing import Dict, List, Optional

from qdrant_client import QdrantClient

from backend.config import (
    COLLECTION_NAME,
    QDRANT_URL,
    TOP_K,
)

from backend.retrievers.dense import DenseRetriever

_client = None


def get_client() -> QdrantClient:

    global _client

    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)

    return _client


_dense = None


def get_dense_retriever():

    global _dense

    if _dense is None:
        _dense = DenseRetriever(get_client())

    return _dense


def retrieve_similar_chunks(
    query_vector: List[float],
    top_k: Optional[int] = None,
    score_threshold: float = 0.0,
    source_filter: Optional[str] = None,
    user_id: Optional[str] = None,
) -> List[Dict]:

    k = top_k or int(os.getenv("TOP_K", TOP_K))

    return get_dense_retriever().search(
        query_vector=query_vector,
        top_k=k,
        score_threshold=score_threshold,
        source_filter=source_filter,
        user_id=user_id,
    )