"""
NexusMind v2 — Hybrid Retrieval (Semantic + BM25 → RRF)
Updated: document-level access control via user_id filter.
"""

import os
from typing import List, Dict, Tuple, Optional

from backend.retrieval  import retrieve_similar_chunks
from backend.bm25_index import bm25_search

RRF_K = 60


def reciprocal_rank_fusion(ranked_lists, id_key="chunk_index", source_key="source", k=RRF_K):
    rrf_scores = {}
    chunk_store = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            uid = (chunk.get(source_key, "?"), chunk.get(id_key, rank))
            rrf_scores[uid]  = rrf_scores.get(uid, 0.0) + 1.0 / (k + rank)
            chunk_store[uid] = chunk
    sorted_uids = sorted(rrf_scores, key=lambda u: -rrf_scores[u])
    merged = []
    for uid in sorted_uids:
        chunk = dict(chunk_store[uid])
        chunk["rrf_score"] = round(rrf_scores[uid], 6)
        merged.append(chunk)
    return merged


def hybrid_retrieve(query, query_vector, top_k=None, use_bm25=True, user_id=None):
    """
    Full hybrid retrieval pipeline with optional user-scoped filtering.
    user_id: when set, only returns chunks owned by this user (access control).
    """
    k = top_k or int(os.environ.get("TOP_K", "5"))
    fetch_k = max(k * 3, 15)

    semantic_results = retrieve_similar_chunks(query_vector, top_k=fetch_k, user_id=user_id)

    bm25_results = []
    if use_bm25:
        raw_bm25 = bm25_search(query, top_k=fetch_k)
        bm25_results = [c for c in raw_bm25 if not user_id or c.get("user_id") == user_id]

    if use_bm25 and bm25_results:
        fused = reciprocal_rank_fusion([semantic_results, bm25_results])
        fusion_method = "Hybrid (Semantic + BM25 → RRF)"
    else:
        fused = semantic_results
        fusion_method = "Semantic only" if not use_bm25 else "Semantic only (BM25 not ready)"

    return {
        "results":       fused[:k],
        "semantic_hits": len(semantic_results),
        "bm25_hits":     len(bm25_results),
        "fusion_method": fusion_method,
    }
