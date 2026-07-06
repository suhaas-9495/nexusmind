"""
NexusMind v2 — Evaluation Test Dataset + Runner
Hardcoded Q&A pairs used to benchmark BM25 vs Dense vs Hybrid.
Run: python -m evaluation.runner
"""

import json
import logging
from pathlib import Path
from typing import List, Dict

from evaluation.metrics import RetrievalEvaluator

logger = logging.getLogger(__name__)

# ── Test Dataset ──────────────────────────────────────────────────────────────
# Format: question + relevant_chunk_ids (source:chunk_index pairs)
# Add your own document-specific Q&A pairs here.

EVAL_DATASET: List[Dict] = [
    {
        "question": "What is the main methodology used in this research?",
        "relevant_chunk_ids": ["doc1:0", "doc1:1"],
        "notes": "Tests broad semantic understanding"
    },
    {
        "question": "What are the key findings or results?",
        "relevant_chunk_ids": ["doc1:3", "doc1:4"],
        "notes": "Tests result extraction"
    },
    {
        "question": "What limitations does the study acknowledge?",
        "relevant_chunk_ids": ["doc1:7"],
        "notes": "Tests specific section retrieval"
    },
    {
        "question": "Who are the authors and what institution are they from?",
        "relevant_chunk_ids": ["doc1:0"],
        "notes": "Tests named entity / BM25 strength"
    },
    {
        "question": "What dataset or data source was used?",
        "relevant_chunk_ids": ["doc1:2"],
        "notes": "Tests keyword-heavy query"
    },
    {
        "question": "What future work is proposed?",
        "relevant_chunk_ids": ["doc1:8", "doc1:9"],
        "notes": "Tests end-section retrieval"
    },
    {
        "question": "How does this compare to previous approaches?",
        "relevant_chunk_ids": ["doc1:1", "doc1:2"],
        "notes": "Tests comparative reasoning"
    },
    {
        "question": "What evaluation metrics are used?",
        "relevant_chunk_ids": ["doc1:5"],
        "notes": "Tests technical term matching"
    },
]


def make_bm25_retriever(top_k: int = 10):
    """Factory for BM25-only retrieval fn."""
    from backend.bm25_index import bm25_search

    def retrieve(query: str) -> List[str]:
        results = bm25_search(query, top_k=top_k)
        return [f"{r.get('source', 'unknown')}:{r.get('chunk_index', 0)}" for r in results]

    return retrieve


def make_dense_retriever(top_k: int = 10):
    """Factory for dense/semantic-only retrieval fn."""
    from backend.embedding_cache import get_cached_embedding
    from backend.retrieval import retrieve_similar_chunks

    def retrieve(query: str) -> List[str]:
        vec = get_cached_embedding(query)
        results = retrieve_similar_chunks(vec, top_k=top_k)
        return [f"{r.get('source', 'unknown')}:{r.get('chunk_index', 0)}" for r in results]

    return retrieve


def make_hybrid_retriever(top_k: int = 10):
    """Factory for hybrid (BM25 + Dense → RRF) retrieval fn."""
    from backend.embedding_cache import get_cached_embedding
    from backend.hybrid_retrieval import hybrid_retrieve

    def retrieve(query: str) -> List[str]:
        vec = get_cached_embedding(query)
        res = hybrid_retrieve(query, vec, top_k=top_k)
        return [f"{r.get('source', 'unknown')}:{r.get('chunk_index', 0)}" for r in res["results"]]

    return retrieve


def run_evaluation(
    dataset: List[Dict] = None,
    k_values: List[int] = None,
    output_path: str = "logs/eval_results.json",
) -> Dict:
    """
    Run full retrieval evaluation: BM25 vs Dense vs Hybrid.
    Saves results to JSON.
    """
    dataset   = dataset   or EVAL_DATASET
    k_values  = k_values  or [1, 3, 5, 10]
    evaluator = RetrievalEvaluator(k_values=k_values)

    print("=" * 60)
    print("NexusMind v2 — Retrieval Evaluation")
    print("=" * 60)
    print(f"Dataset size: {len(dataset)} queries")
    print(f"k values: {k_values}")
    print()

    results = evaluator.compare_strategies(
        dataset    = dataset,
        bm25_fn    = make_bm25_retriever(),
        dense_fn   = make_dense_retriever(),
        hybrid_fn  = make_hybrid_retriever(),
    )

    # Pretty print comparison table
    strategies = ["bm25", "dense", "hybrid"]
    metrics    = ["mrr", "map"] + [f"recall@{k}" for k in k_values]

    print(f"\n{'Metric':<15}", end="")
    for s in strategies:
        print(f"{s.upper():>14}", end="")
    print(f"  {'Winner':<12}")
    print("-" * 65)

    for m in metrics:
        print(f"{m:<15}", end="")
        for s in strategies:
            val = results[s].get(m, 0)
            print(f"{val:>14.4f}", end="")
        winner = results.get(f"winner_{m}", "-")
        print(f"  {winner.upper():<12}")

    print()
    print(f"Avg Latency (ms):", end="")
    for s in strategies:
        print(f"{results[s]['avg_latency_ms']:>14.1f}", end="")
    print()

    # Save to disk
    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        # Remove per_query detail for cleaner output file (keep it if you want)
        summary = {k: {m: v for m, v in v.items() if m != "per_query"}
                   if isinstance(v, dict) else v
                   for k, v in results.items()}
        json.dump(summary, f, indent=2)

    print(f"\n✅ Results saved to {output_path}")
    return results


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    run_evaluation()
