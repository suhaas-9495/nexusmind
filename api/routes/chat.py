"""
NexusMind v2 — /chat endpoint
Multi-turn conversational RAG with per-user session memory.
"""

import time
import logging
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from auth.jwt_handler import get_current_user
from agent.runner import run_agent
from agent.memory import ConversationMemory
from backend.embedding import embed_texts
from backend.hybrid_retrieval import hybrid_retrieve
from backend.llm_answer import generate_answer
from backend.embedding_cache import get_cached_embedding
from observability.metrics import record_query

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory session store: user_id → ConversationMemory
# In production, replace with Redis for multi-instance support
_sessions: dict[str, ConversationMemory] = {}


def _get_session(user_id: str) -> ConversationMemory:
    if user_id not in _sessions:
        _sessions[user_id] = ConversationMemory(max_turns=10)
    return _sessions[user_id]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)
    mode: str = Field("rag", pattern="^(rag|agent)$")
    use_hybrid: bool = True


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    mode: str
    agent_steps: Optional[List[str]] = None
    latency_ms: float
    history: List[ChatMessage]


@router.post("/", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Multi-turn chat endpoint. Each user has a persistent session (in-process).
    mode=rag → hybrid retrieval + LLM
    mode=agent → full agentic planning loop
    """
    start = time.perf_counter()
    user_id = current_user["sub"]
    memory = _get_session(user_id)

    try:
        memory.add("user", req.message)

        if req.mode == "agent":
            result = run_agent(req.message, memory)
            answer = result["final_answer"]
            sources = result.get("sources", [])
            agent_steps = result.get("steps", [])
        else:
            query_vec = get_cached_embedding(req.message)
            hybrid_res = hybrid_retrieve(
                query=req.message,
                query_vector=query_vec,
                top_k=req.top_k,
                use_bm25=req.use_hybrid,
                user_id=user_id,
            )
            chunks = hybrid_res["results"]
            answer = generate_answer(req.message, chunks, memory.get_context())
            sources = chunks
            agent_steps = []

        memory.add("assistant", answer)

        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        record_query(user_id=user_id, latency_ms=latency_ms, chunks_returned=len(sources), fusion=req.mode)

        return ChatResponse(
            answer=answer,
            sources=sources,
            mode=req.mode,
            agent_steps=agent_steps,
            latency_ms=latency_ms,
            history=[ChatMessage(role=t["role"], content=t["content"]) for t in memory.get_all()],
        )

    except Exception as e:
        logger.error(f"Chat error for user={user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session")
async def clear_session(current_user: dict = Depends(get_current_user)):
    """Clear the current user's conversation history."""
    _sessions.pop(current_user["sub"], None)
    return {"message": "Session cleared."}
