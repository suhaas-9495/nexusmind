"""
NexusMind v2 — Retrieval Evaluation
Implements Recall@k and MRR metrics.
Compares BM25 vs Dense vs Hybrid (RRF) retrieval strategies.
"""

import time
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


# ── Core Metrics ──────────────────────────────────────────────────────────────

def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Recall@k = |relevant ∩ top-k retrieved| / |relevant|
    Measures: of all relevant docs, how many did we find in top-k?
    """
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(top_k & relevant) / len(relevant)


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """
    Reciprocal Rank = 1 / rank_of_first_relevant_doc
    Returns 0.0 if no relevant doc found.
    """
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(results: List[Tuple[List[str], List[str]]]) -> float:
    """
    MRR = mean of reciprocal ranks across all queries.
    results: list of (retrieved_ids, relevant_ids) tuples.
    """
    if not results:
        return 0.0
    rr_scores = [reciprocal_rank(ret, rel) for ret, rel in results]
    return sum(rr_scores) / len(rr_scores)


def average_precision(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """
    Average Precision (AP) — area under precision-recall curve.
    """
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    hits = 0
    sum_prec = 0.0
    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            hits += 1
            sum_prec += hits / i
    return sum_prec / len(relevant) if hits > 0 else 0.0


def mean_average_precision(results: List[Tuple[List[str], List[str]]]) -> float:
    """MAP = mean AP across all queries."""
    if not results:
        return 0.0
    return sum(average_precision(ret, rel) for ret, rel in results) / len(results)


# ── Evaluator ─────────────────────────────────────────────────────────────────

class RetrievalEvaluator:
    """
    Evaluates retrieval strategies (BM25, Dense, Hybrid) on a Q&A test dataset.
    """

    def __init__(self, k_values: List[int] = None):
        self.k_values = k_values or [1, 3, 5, 10]

    def evaluate_single(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
    ) -> Dict:
        """Compute all metrics for one query."""
        result = {
            "rr":   reciprocal_rank(retrieved_ids, relevant_ids),
            "ap":   average_precision(retrieved_ids, relevant_ids),
        }
        for k in self.k_values:
            result[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
        return result

    def evaluate_dataset(
        self,
        dataset: List[Dict],
        retrieval_fn,
        strategy_name: str,
    ) -> Dict:
        """
        Run evaluation over full test dataset.

        Args:
            dataset: list of {"question": str, "relevant_chunk_ids": list[str]}
            retrieval_fn: callable(query: str) → list[str] (ordered chunk ids)
            strategy_name: "bm25" | "dense" | "hybrid"

        Returns:
            Aggregated metric dict.
        """
        all_results = []
        latencies   = []
        per_query   = []

        for item in dataset:
            question    = item["question"]
            relevant    = item["relevant_chunk_ids"]

            start = time.perf_counter()
            retrieved = retrieval_fn(question)
            latency_ms = round((time.perf_counter() - start) * 1000, 1)

            metrics = self.evaluate_single(retrieved, relevant)
            metrics["question"]   = question
            metrics["latency_ms"] = latency_ms
            per_query.append(metrics)

            all_results.append((retrieved, relevant))
            latencies.append(latency_ms)

        # Aggregate
        agg = {
            "strategy":       strategy_name,
            "num_queries":    len(dataset),
            "mrr":            round(mean_reciprocal_rank(all_results), 4),
            "map":            round(mean_average_precision(all_results), 4),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
            "per_query":      per_query,
        }
        for k in self.k_values:
            scores = [r[f"recall@{k}"] for r in per_query]
            agg[f"recall@{k}"] = round(sum(scores) / len(scores), 4) if scores else 0.0

        logger.info(
            f"[Eval] {strategy_name}: MRR={agg['mrr']:.3f}, "
            f"Recall@5={agg.get('recall@5', 0):.3f}, "
            f"MAP={agg['map']:.3f}, "
            f"avg_latency={agg['avg_latency_ms']}ms"
        )
        return agg

    def compare_strategies(
        self,
        dataset: List[Dict],
        bm25_fn,
        dense_fn,
        hybrid_fn,
    ) -> Dict:
        """
        Run all three strategies and return comparison table.
        """
        results = {
            "bm25":   self.evaluate_dataset(dataset, bm25_fn,   "BM25"),
            "dense":  self.evaluate_dataset(dataset, dense_fn,  "Dense"),
            "hybrid": self.evaluate_dataset(dataset, hybrid_fn, "Hybrid (RRF)"),
        }

        # Identify winner per metric
        for metric in ["mrr", "map"] + [f"recall@{k}" for k in self.k_values]:
            scores = {s: results[s].get(metric, 0) for s in results}
            winner = max(scores, key=scores.get)
            results[f"winner_{metric}"] = winner

        return results
