"""
NexusMind – RAG Tool
Used by the agent executor to retrieve relevant document chunks.
Uses hybrid retrieval (Semantic + BM25 → RRF) automatically.
"""

import os
from typing import Tuple, List, Dict

from backend.embedding         import embed_texts
from backend.hybrid_retrieval  import hybrid_retrieve
from backend.config            import TOP_K


def rag_retrieve(query: str) -> Tuple[str, List[Dict]]:
    """
    Embed `query`, run hybrid retrieval (semantic + BM25 fused via RRF),
    return (text_summary, chunks).
    """
    if not query.strip():
        return "Empty query — nothing retrieved.", []

    k = int(os.environ.get("TOP_K", TOP_K))

    try:
        vec    = embed_texts([query])[0]
        result = hybrid_retrieve(query, vec, top_k=k)
        chunks = result["results"]
    except Exception as e:
        return f"Retrieval error: {e}", []

    if not chunks:
        return "No relevant chunks found for this query.", []

    fusion = result.get("fusion_method", "hybrid")
    lines  = [f"Retrieved {len(chunks)} chunks via [{fusion}] for: '{query}'\n"]

    for i, ch in enumerate(chunks, 1):
        rrf   = ch.get("rrf_score", 0)
        sem   = ch.get("score", 0)
        bm25  = ch.get("bm25_score", 0)
        lines.append(
            f"[{i}] RRF={rrf:.4f} | Semantic={sem:.3f} | BM25={bm25:.3f} | "
            f"file={ch.get('source','?')} chunk#{ch.get('chunk_index','?')}\n"
            f"{ch.get('chunk_text', '')[:300]}…"
        )

    return "\n\n".join(lines), chunks
