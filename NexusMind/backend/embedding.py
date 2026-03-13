"""
NexusMind – Embedding
Uses Ollama's local embedding model (default: nomic-embed-text).
"""

import os
import requests
from typing import List

from backend.config import OLLAMA_BASE_URL, EMBED_MODEL


def embed_texts(texts: List[str], model: str | None = None) -> List[List[float]]:
    """
    Embed a list of texts using Ollama.
    Returns a list of float vectors, one per input text.
    """
    model = model or os.environ.get("EMBED_MODEL", EMBED_MODEL)
    url   = f"{OLLAMA_BASE_URL}/api/embeddings"

    embeddings: List[List[float]] = []

    for text in texts:
        if not text.strip():
            # Return zero vector for empty strings (Qdrant needs consistent dims)
            embeddings.append([0.0] * 768)
            continue

        payload = {"model": model, "prompt": text}
        resp    = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        embeddings.append(resp.json()["embedding"])

    return embeddings


def embed_query(query: str, model: str | None = None) -> List[float]:
    """Convenience wrapper for a single query string."""
    return embed_texts([query], model=model)[0]
