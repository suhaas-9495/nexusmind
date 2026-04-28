"""
NexusMind v2 — /query endpoint
Single-turn RAG query with hybrid retrieval + latency tracking.
"""

import time
import logging
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from auth.jwt_handler import get_current_user
from backend.embedding import embed_texts
from backend.hybrid_retrieval import hybrid_retrieve
from backend.llm_answer import generate_answer
from backend.embedding_cache import get_cached_embedding
from observability.metrics import record_query
from observability.logger import get_request_logger

router = APIRouter()
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)
    use_hybrid: bool = True
    source_filter: Optional[str] = None


class SourceChunk(BaseModel):
    source: str
    chunk_index: int
    chunk_text: str
    score: Optional[float] = None
    rrf_score: Optional[float] = None
    bm25_score: Optional[float] = None
    user_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    fusion_method: str
    semantic_hits: int
    bm25_hits: int
    latency_ms: float
    token_estimate: int


@router.post("/", response_model=QueryResponse)
async def query_documents(
    req: QueryRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    RAG query with hybrid retrieval (BM25 + semantic → RRF).
    Results are filtered to documents owned by the current user.
    """
    start = time.perf_counter()

    try:
        # ── Embedding (with cache) ────────────────────────────────────────
        query_vec = get_cached_embedding(req.query)

        # ── Hybrid retrieval ──────────────────────────────────────────────
        hybrid_res = hybrid_retrieve(
            query=req.query,
            query_vector=query_vec,
            top_k=req.top_k,
            use_bm25=req.use_hybrid,
            user_id=current_user["sub"],   # document-level access control
        )
        chunks = hybrid_res["results"]

        # ── LLM answer ────────────────────────────────────────────────────
        answer = generate_answer(req.query, chunks)

        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        token_estimate = len(answer.split()) + sum(len(c.get("chunk_text","").split()) for c in chunks)

        record_query(
            user_id=current_user["sub"],
            latency_ms=latency_ms,
            chunks_returned=len(chunks),
            fusion=hybrid_res["fusion_method"],
        )

        get_request_logger().info(
            f"User={current_user['sub']} query='{req.query[:60]}' "
            f"→ {len(chunks)} chunks, {latency_ms}ms"
        )

        return QueryResponse(
            answer=answer,
            sources=[SourceChunk(**{k: v for k, v in c.items() if k in SourceChunk.__fields__}) for c in chunks],
            fusion_method=hybrid_res["fusion_method"],
            semantic_hits=hybrid_res["semantic_hits"],
            bm25_hits=hybrid_res["bm25_hits"],
            latency_ms=latency_ms,
            token_estimate=token_estimate,
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
