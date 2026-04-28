"""
NexusMind v2 — Unit Tests for Evaluation Metrics
Run: pytest tests/test_evaluation.py -v
"""

import pytest
from evaluation.metrics import (
    recall_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
    average_precision,
    mean_average_precision,
    RetrievalEvaluator,
)


class TestRecallAtK:
    def test_perfect_recall(self):
        assert recall_at_k(["a", "b", "c"], ["a", "b"], k=3) == 1.0

    def test_zero_recall(self):
        assert recall_at_k(["x", "y", "z"], ["a", "b"], k=3) == 0.0

    def test_partial_recall(self):
        assert recall_at_k(["a", "x", "b"], ["a", "b", "c"], k=3) == pytest.approx(2/3)

    def test_recall_at_1(self):
        assert recall_at_k(["a", "b", "c"], ["a"], k=1) == 1.0
        assert recall_at_k(["b", "a", "c"], ["a"], k=1) == 0.0

    def test_empty_relevant(self):
        assert recall_at_k(["a", "b"], [], k=3) == 0.0


class TestReciprocalRank:
    def test_first_hit(self):
        assert reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0

    def test_second_hit(self):
        assert reciprocal_rank(["x", "a", "b"], ["a"]) == pytest.approx(0.5)

    def test_third_hit(self):
        assert reciprocal_rank(["x", "y", "a"], ["a"]) == pytest.approx(1/3)

    def test_no_hit(self):
        assert reciprocal_rank(["x", "y", "z"], ["a"]) == 0.0

    def test_multiple_relevant(self):
        # Should return RR of FIRST relevant doc
        assert reciprocal_rank(["x", "a", "b"], ["a", "b"]) == pytest.approx(0.5)


class TestMRR:
    def test_perfect_mrr(self):
        results = [
            (["a", "b"], ["a"]),
            (["a", "b"], ["a"]),
        ]
        assert mean_reciprocal_rank(results) == 1.0

    def test_mixed_mrr(self):
        results = [
            (["a", "b"], ["a"]),   # RR = 1.0
            (["x", "a"], ["a"]),   # RR = 0.5
        ]
        assert mean_reciprocal_rank(results) == pytest.approx(0.75)

    def test_empty(self):
        assert mean_reciprocal_rank([]) == 0.0


class TestAveragePrecision:
    def test_perfect_ap(self):
        assert average_precision(["a", "b"], ["a", "b"]) == 1.0

    def test_zero_ap(self):
        assert average_precision(["x", "y"], ["a", "b"]) == 0.0

    def test_partial_ap(self):
        # Relevant: a, b. Retrieved: a at 1, x at 2, b at 3
        # P@1=1.0 (hit a), P@3=2/3 (hits a,b) → AP = (1.0 + 2/3) / 2
        val = average_precision(["a", "x", "b"], ["a", "b"])
        assert val == pytest.approx((1.0 + 2/3) / 2)


class TestEvaluator:
    def test_single_query(self):
        ev = RetrievalEvaluator(k_values=[1, 3, 5])
        metrics = ev.evaluate_single(["a", "b", "c"], ["a", "c"])
        assert "recall@1" in metrics
        assert "recall@3" in metrics
        assert "rr" in metrics
        assert metrics["recall@3"] == 1.0

    def test_dataset_eval(self):
        ev = RetrievalEvaluator(k_values=[3])
        dataset = [
            {"question": "q1", "relevant_chunk_ids": ["a"]},
            {"question": "q2", "relevant_chunk_ids": ["b"]},
        ]
        # Mock retrieval fn: always returns ["a", "b", "c"]
        result = ev.evaluate_dataset(dataset, lambda q: ["a", "b", "c"], "test")
        assert result["strategy"] == "test"
        # q1: relevant=["a"]→RR=1.0  q2: relevant=["b"]→RR=0.5  MRR=0.75
        assert result["mrr"] == pytest.approx(0.75)
        assert result["recall@3"] == 1.0
