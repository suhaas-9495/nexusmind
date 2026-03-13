"""
NexusMind – Hybrid Retrieval (Semantic + BM25 → RRF Fusion)

Why hybrid?
  • Semantic search  → understands meaning, synonyms, paraphrasing
  • BM25             → excels at exact keywords, rare terms, named entities
  • Together via RRF → consistently beats either alone (industry standard)

Reciprocal Rank Fusion (RRF):
  score(d) = Σ  1 / (k + rank_i(d))
  where k=60 (default), rank_i is position in each ranked list.
  RRF is parameter-robust and requires NO score normalisation.
"""

import os
from typing import List, Dict, Tuple

from backend.retrieval  import retrieve_similar_chunks
from backend.bm25_index import bm25_search
from backend.config     import TOP_K

# RRF constant — 60 is the well-tested default from the original paper
RRF_K = 60


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict]],
    id_key:       str = "chunk_index",
    source_key:   str = "source",
    k:            int = RRF_K,
) -> List[Dict]:
    """
    Merge N ranked lists using RRF.

    Args:
        ranked_lists: Each inner list is already sorted best-first.
        id_key:       Payload key used to identify unique chunks.
        source_key:   Secondary key used with id_key for uniqueness.
        k:            RRF smoothing constant.

    Returns:
        Merged list sorted by descending RRF score, with 'rrf_score' added.
    """
    rrf_scores: Dict[Tuple, float] = {}
    chunk_store: Dict[Tuple, Dict] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            uid = (chunk.get(source_key, "?"), chunk.get(id_key, rank))
            rrf_scores[uid]  = rrf_scores.get(uid, 0.0) + 1.0 / (k + rank)
            chunk_store[uid] = chunk   # keep latest copy

    # Sort by RRF score descending
    sorted_uids = sorted(rrf_scores, key=lambda u: -rrf_scores[u])

    merged = []
    for uid in sorted_uids:
        chunk = dict(chunk_store[uid])
        chunk["rrf_score"] = round(rrf_scores[uid], 6)
        merged.append(chunk)

    return merged


# ── Main hybrid retrieval entry point ────────────────────────────────────────

def hybrid_retrieve(
    query:          str,
    query_vector:   List[float],
    top_k:          int | None = None,
    semantic_weight: float = 0.6,   # for display only (RRF doesn't need weights)
    use_bm25:       bool = True,
) -> Dict:
    """
    Full hybrid retrieval pipeline.

    Args:
        query:           Raw query string (for BM25).
        query_vector:    Embedded query vector (for semantic search).
        top_k:           Final number of results to return.
        semantic_weight: Fraction of results to fetch from semantic (hint only).
        use_bm25:        Toggle BM25 off to fall back to pure semantic.

    Returns:
        {
            "results":         list[dict],   # merged, ranked chunks
            "semantic_hits":   int,
            "bm25_hits":       int,
            "fusion_method":   str,
        }
    """
    k = top_k or int(os.environ.get("TOP_K", TOP_K))

    # Fetch more candidates than needed so fusion has room to work
    fetch_k = max(k * 3, 15)

    # ── Semantic retrieval ────────────────────────────────────────────────────
    semantic_results = retrieve_similar_chunks(query_vector, top_k=fetch_k)

    # ── BM25 retrieval ────────────────────────────────────────────────────────
    bm25_results = []
    if use_bm25:
        bm25_results = bm25_search(query, top_k=fetch_k)

    # ── Fusion ────────────────────────────────────────────────────────────────
    if use_bm25 and bm25_results:
        ranked_lists = [semantic_results, bm25_results]
        fused        = reciprocal_rank_fusion(ranked_lists)
        fusion_method = "Hybrid (Semantic + BM25 → RRF)"
    else:
        # BM25 index not ready yet — fall back gracefully
        fused         = semantic_results
        fusion_method = "Semantic only (BM25 index not ready)"

    # Trim to final top_k
    final = fused[:k]

    return {
        "results":       final,
        "semantic_hits": len(semantic_results),
        "bm25_hits":     len(bm25_results),
        "fusion_method": fusion_method,
    }
