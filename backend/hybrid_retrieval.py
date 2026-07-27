"""
NexusMind - Production Hybrid Retrieval
"""

import os
from typing import Optional

from backend.models.retrieval_response import RetrievalResponse
from backend.retrieval import retrieve_similar_chunks
from backend.retrievers.fusion import ReciprocalRankFusion
from backend.retrievers.reranker import CrossEncoderReranker
from backend.retrievers.sparse import SparseRetriever


_sparse = SparseRetriever()
_fusion = ReciprocalRankFusion()
_reranker = CrossEncoderReranker()


class HybridRetriever:

    def retrieve(
        self,
        query: str,
        query_vector,
        top_k: Optional[int] = None,
        use_dense: bool = True,
        use_sparse: bool = True,
        user_id: Optional[str] = None,
    ) -> RetrievalResponse:

        k = top_k or int(os.getenv("TOP_K", "5"))
        fetch_k = max(k * 3, 15)

        dense_results = []
        sparse_results = []

        if use_dense:
            dense_results = retrieve_similar_chunks(
                query_vector=query_vector,
                top_k=fetch_k,
                user_id=user_id,
            )

        if use_sparse:
            sparse_results = _sparse.search(
                query=query,
                top_k=fetch_k,
                user_id=user_id,
            )

        # -------------------------
        # Hybrid Retrieval
        # -------------------------

        if dense_results and sparse_results:

            results = _fusion.fuse(
                ranked_lists=[
                    dense_results,
                    sparse_results,
                ],
                weights=[
                    0.7,
                    0.3,
                ],
            )

            results = _reranker.rerank(
                query=query,
                documents=results,
                top_k=k,
            )

            method = "Hybrid + RRF + CrossEncoder"

        # -------------------------
        # Dense Only
        # -------------------------

        elif dense_results:

            results = _reranker.rerank(
                query=query,
                documents=dense_results,
                top_k=k,
            )

            method = "Dense + CrossEncoder"

        # -------------------------
        # BM25 Only
        # -------------------------

        elif sparse_results:

            results = _reranker.rerank(
                query=query,
                documents=sparse_results,
                top_k=k,
            )

            method = "BM25 + CrossEncoder"

        # -------------------------
        # No Results
        # -------------------------

        else:

            results = []
            method = "No Retrieval"

        return RetrievalResponse(
            results=results,
            semantic_hits=len(dense_results),
            bm25_hits=len(sparse_results),
            fusion_method=method,
        )


_hybrid = HybridRetriever()


def hybrid_retrieve(
    query,
    query_vector,
    top_k=None,
    use_bm25=True,
    user_id=None,
):
    return _hybrid.retrieve(
        query=query,
        query_vector=query_vector,
        top_k=top_k,
        use_dense=True,
        use_sparse=use_bm25,
        user_id=user_id,
    )