"""
NexusMind – LLM Answer Generation
Generates grounded, cited answers using Ollama chat API.
Supports both blocking and streaming modes.
"""

import os
import json
import requests
from typing import List, Dict, Generator

from backend.config import OLLAMA_BASE_URL, LLM_MODEL, TEMPERATURE


def _get_model() -> str:
    return os.environ.get("LLM_MODEL", LLM_MODEL)


def _build_system_prompt() -> str:
    return (
        "You are NexusMind, an intelligent document assistant. "
        "Answer the user's question using ONLY the provided context chunks. "
        "Always cite sources using [Source N] notation inline. "
        "If the answer cannot be found in the context, say so clearly — do NOT hallucinate. "
        "Be concise, structured, and helpful. Use markdown formatting when it aids clarity."
    )


def _build_context_block(chunks: List[Dict]) -> str:
    if not chunks:
        return "No context available."

    blocks = []
    for i, chunk in enumerate(chunks, 1):
        blocks.append(
            f"[Source {i}]\n"
            f"File: {chunk.get('source', 'unknown')}\n"
            f"Chunk #{chunk.get('chunk_index', '?')}\n"
            f"Relevance Score: {chunk.get('score', 0):.3f}\n\n"
            f"{chunk.get('chunk_text', '').strip()}"
        )
    return "\n\n---\n\n".join(blocks)


def generate_answer(
    question: str,
    chunks:   List[Dict],
    conversation_context: str = "",
) -> str:
    """
    Blocking answer generation.

    Args:
        question:             User's question.
        chunks:               Retrieved context chunks.
        conversation_context: Prior conversation turns for multi-turn coherence.

    Returns:
        LLM-generated answer string.
    """
    context_text = _build_context_block(chunks)

    messages = [{"role": "system", "content": _build_system_prompt()}]

    if conversation_context:
        messages.append({
            "role": "system",
            "content": f"Previous conversation context:\n{conversation_context}"
        })

    user_content = (
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Provide a well-structured answer with source citations:"
    )
    messages.append({"role": "user", "content": user_content})

    payload = {
        "model":   _get_model(),
        "messages": messages,
        "stream":  False,
        "options": {"temperature": float(os.environ.get("TEMPERATURE", TEMPERATURE))},
    }

    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def generate_answer_stream(
    question: str,
    chunks:   List[Dict],
    conversation_context: str = "",
) -> Generator[str, None, None]:
    """
    Streaming answer generation — yields text tokens as they arrive.
    """
    context_text = _build_context_block(chunks)

    messages = [{"role": "system", "content": _build_system_prompt()}]

    if conversation_context:
        messages.append({
            "role": "system",
            "content": f"Previous conversation:\n{conversation_context}"
        })

    messages.append({
        "role": "user",
        "content": (
            f"Context:\n{context_text}\n\n"
            f"Question: {question}\n\n"
            "Provide a well-structured answer with source citations:"
        ),
    })

    payload = {
        "model":    _get_model(),
        "messages": messages,
        "stream":   True,
        "options":  {"temperature": float(os.environ.get("TEMPERATURE", TEMPERATURE))},
    }

    with requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        stream=True,
        timeout=180,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token
                if data.get("done"):
                    break
