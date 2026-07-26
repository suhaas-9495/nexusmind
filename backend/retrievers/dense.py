from typing import Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
)

from backend.config import COLLECTION_NAME


class DenseRetriever:
    def __init__(self, client: QdrantClient):
        self.client = client

    def search(
        self,
        query_vector: List[float],
        top_k: int,
        score_threshold: float = 0.0,
        source_filter: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict]:

        must = []

        if source_filter:
            must.append(
                FieldCondition(
                    key="source",
                    match=MatchValue(value=source_filter),
                )
            )

        if user_id:
            must.append(
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id),
                )
            )

        query_filter = Filter(must=must) if must else None

        hits = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
        ).points

        results = []

        for hit in hits:

            if hit.score < score_threshold:
                continue

            payload = dict(hit.payload or {})

            payload["dense_score"] = float(hit.score)

            results.append(payload)

        return results