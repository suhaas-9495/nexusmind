"""
NexusMind v2 — Embedding Cache
Local in-memory + disk-backed cache for query embeddings.
Avoids redundant Ollama embedding calls for repeated queries.

For production with multiple workers, swap to Redis:
    import redis; r = redis.Redis(); r.set(key, json.dumps(vec))
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# In-memory cache: query_hash → (vector, timestamp)
_cache: Dict[str, tuple] = {}
CACHE_TTL_SECS = 3600   # 1 hour
DISK_CACHE_PATH = Path("logs/embedding_cache.json")
MAX_CACHE_SIZE  = 1000  # evict oldest when full


def _hash_query(query: str) -> str:
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]


def _evict_if_needed() -> None:
    if len(_cache) >= MAX_CACHE_SIZE:
        # Evict oldest 20%
        sorted_keys = sorted(_cache, key=lambda k: _cache[k][1])
        for k in sorted_keys[:MAX_CACHE_SIZE // 5]:
            del _cache[k]
        logger.debug(f"Evicted {MAX_CACHE_SIZE // 5} cache entries.")


def get_cached_embedding(query: str) -> List[float]:
    """
    Return cached embedding or compute + cache it.
    """
    from backend.embedding import embed_texts  # lazy import to avoid circular

    key = _hash_query(query)
    now = time.time()

    if key in _cache:
        vec, ts = _cache[key]
        if now - ts < CACHE_TTL_SECS:
            logger.debug(f"Embedding cache HIT for '{query[:40]}'")
            return vec
        else:
            del _cache[key]  # expired

    # Compute embedding
    logger.debug(f"Embedding cache MISS for '{query[:40]}'")
    vec = embed_texts([query])[0]

    _evict_if_needed()
    _cache[key] = (vec, now)

    return vec


def get_cache_stats() -> dict:
    return {
        "size": len(_cache),
        "max_size": MAX_CACHE_SIZE,
        "ttl_secs": CACHE_TTL_SECS,
    }
