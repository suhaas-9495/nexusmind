"""
NexusMind v2 — Metrics Tracker
Tracks latency, token usage, and failure cases.
Stores to JSON for persistence and export.
"""

import json
import time
import logging
from pathlib import Path
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)

METRICS_PATH = Path("logs/metrics.jsonl")
METRICS_PATH.parent.mkdir(exist_ok=True)

# In-memory aggregates
_query_latencies: list[float] = []
_upload_counts: int = 0
_query_counts: int = 0
_error_counts: int = 0
_token_totals: int = 0


def _append_event(event: dict) -> None:
    event["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        with open(METRICS_PATH, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        logger.warning(f"Metrics write failed: {e}")


def record_upload(filename: str, chunks: int, elapsed: float) -> None:
    global _upload_counts
    _upload_counts += 1
    _append_event({
        "event":    "upload",
        "filename": filename,
        "chunks":   chunks,
        "elapsed_sec": elapsed,
    })


def record_query(
    user_id: str,
    latency_ms: float,
    chunks_returned: int,
    fusion: str,
    tokens: int = 0,
) -> None:
    global _query_counts, _token_totals
    _query_counts += 1
    _token_totals += tokens
    _query_latencies.append(latency_ms)
    _append_event({
        "event":          "query",
        "user_id":        user_id,
        "latency_ms":     latency_ms,
        "chunks_returned": chunks_returned,
        "fusion":         fusion,
        "tokens":         tokens,
    })


def record_error(endpoint: str, error: str, user_id: Optional[str] = None) -> None:
    global _error_counts
    _error_counts += 1
    _append_event({
        "event":    "error",
        "endpoint": endpoint,
        "error":    str(error)[:500],
        "user_id":  user_id,
    })


def get_summary() -> dict:
    latencies = _query_latencies or [0]
    return {
        "total_queries":    _query_counts,
        "total_uploads":    _upload_counts,
        "total_errors":     _error_counts,
        "total_tokens":     _token_totals,
        "avg_latency_ms":   round(sum(latencies) / len(latencies), 1),
        "p95_latency_ms":   round(sorted(latencies)[int(len(latencies) * 0.95)], 1) if len(latencies) > 1 else latencies[0],
        "max_latency_ms":   max(latencies),
    }
