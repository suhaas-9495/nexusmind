from typing import Dict, List, Optional

from backend.bm25_index import bm25_search


class SparseRetriever:

    def search(
        self,
        query: str,
        top_k: int,
        user_id: Optional[str] = None,
    ) -> List[Dict]:

        results = bm25_search(query=query, top_k=top_k)

        if user_id:
            results = [
                r
                for r in results
                if r.get("user_id") == user_id
            ]

        return results