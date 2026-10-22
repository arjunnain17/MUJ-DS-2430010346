"""Hand-checked verification of the ranking metrics.

Every assertion below uses values computed by hand, so the metrics are validated
independently of any retrieval model.
"""

from __future__ import annotations

import pytest

from src.metrics import (
    average_precision,
    first_relevant_rank,
    hit_at_k,
    latency_summary,
    macro_average,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    relevant_retrieved,
    summarize_run,
)


class TestPrecisionAtK:
    def test_all_retrieved_relevant(self):
        # 3 of 3 hits relevant -> 3/3 = 1.0
        assert precision_at_k(["A", "B", "C"], ["A", "B", "C"], 3) == pytest.approx(1.0)

    def test_partial_relevance(self):
        # 2 of 3 hits relevant -> 2/3
        assert precision_at_k(["A", "X", "C"], ["A", "C"], 3) == pytest.approx(2 / 3)

    def test_no_relevance(self):
        assert precision_at_k(["X", "Y", "Z"], ["A", "B"], 3) == pytest.approx(0.0)

    def test_denominator_is_k_not_result_count(self):
        # Only one document returned, and it is relevant -> 1/3, not 1/1.
        assert precision_at_k(["A"], ["A"], 3) == pytest.approx(1 / 3)

    def test_extra_results_beyond_k_are_ignored(self):
        # Relevant document sits at rank 4, outside k=3 -> precision 0.
        assert precision_at_k(["X", "Y", "Z", "A"], ["A"], 3) == pytest.approx(0.0)

    def test_k_zero_is_safe(self):
        assert precision_at_k(["A"], ["A"], 0) == pytest.approx(0.0)


class TestRecallAtK:
    def test_full_recall(self):
        # Both relevant documents captured in the top 3 -> 2/2
        assert recall_at_k(["A", "X", "B"], ["A", "B"], 3) == pytest.approx(1.0)

    def test_partial_recall(self):
        # 1 of 2 relevant documents captured -> 0.5
        assert recall_at_k(["A", "X", "Y"], ["A", "B"], 3) == pytest.approx(0.5)

    def test_zero_recall(self):
        assert recall_at_k(["X", "Y", "Z"], ["A"], 3) == pytest.approx(0.0)

    def test_no_relevant_labels_is_zero_not_error(self):
        assert recall_at_k(["A", "B"], [], 3) == pytest.approx(0.0)

    def test_recall_ignores_position_beyond_k(self):
        assert recall_at_k(["X", "Y", "Z", "A"], ["A"], 3) == pytest.approx(0.0)


class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank(["A", "B"], ["A"]) == pytest.approx(1.0)

    def test_second_position(self):
        assert reciprocal_rank(["X", "A"], ["A"]) == pytest.approx(0.5)

    def test_fourth_position(self):
        assert reciprocal_rank(["X", "Y", "Z", "A"], ["A"]) == pytest.approx(0.25)

    def test_no_relevant_result_scores_zero(self):
        assert reciprocal_rank(["X", "Y"], ["A"]) == pytest.approx(0.0)

    def test_earliest_relevant_result_wins(self):
        assert reciprocal_rank(["X", "A", "B"], ["A", "B"]) == pytest.approx(0.5)

    def test_first_relevant_rank_helper(self):
        assert first_relevant_rank(["X", "A"], ["A"]) == 2
        assert first_relevant_rank(["X", "Y"], ["A"]) is None


class TestOtherMetrics:
    def test_hit_at_k(self):
        assert hit_at_k(["A"], ["A"], 3) == 1
        assert hit_at_k(["X"], ["A"], 3) == 0

    def test_average_precision_hand_checked(self):
        # hits at ranks 1 and 3 with 2 relevant -> (1/1 + 2/3) / 2
        expected = (1.0 + (2 / 3)) / 2
        assert average_precision(["A", "X", "B"], ["A", "B"], 3) == pytest.approx(expected)

    def test_relevant_retrieved_preserves_rank_order(self):
        assert relevant_retrieved(["B", "X", "A"], ["A", "B"], 3) == ["B", "A"]

    def test_macro_average(self):
        assert macro_average([1.0, 0.5, 0.0]) == pytest.approx(0.5)
        assert macro_average([]) == pytest.approx(0.0)


class TestLatencySummary:
    def test_mean_and_median(self):
        summary = latency_summary([10.0, 20.0, 30.0, 40.0])
        assert summary["mean_latency_ms"] == pytest.approx(25.0)
        assert summary["median_latency_ms"] == pytest.approx(25.0)
        assert summary["queries"] == 4

    def test_empty_is_zeroed(self):
        summary = latency_summary([])
        assert summary["mean_latency_ms"] == 0.0
        assert summary["median_latency_ms"] == 0.0


class TestSummarizeRun:
    def _rows(self):
        return [
            {
                "precision_at_k": 1 / 3,
                "recall_at_k": 0.5,
                "reciprocal_rank": 1.0,
                "hit_at_k": 1,
                "average_precision": 0.5,
            },
            {
                "precision_at_k": 0.0,
                "recall_at_k": 0.0,
                "reciprocal_rank": 0.0,
                "hit_at_k": 0,
                "average_precision": 0.0,
            },
        ]

    def test_macro_averaged_metrics(self):
        summary = summarize_run(self._rows(), k=3)
        assert summary["evaluated_queries"] == 2
        assert summary["k"] == 3
        assert summary["precision_at_k"] == pytest.approx((1 / 3 + 0.0) / 2)
        assert summary["recall_at_k"] == pytest.approx(0.25)
        assert summary["mrr"] == pytest.approx(0.5)
        assert summary["queries_with_no_relevant_retrieved"] == 1

    def test_latency_is_attached_when_supplied(self):
        summary = summarize_run(self._rows(), k=3, latencies_ms=[5.0, 15.0])
        assert summary["median_latency_ms"] == pytest.approx(10.0)
        assert summary["mean_latency_ms"] == pytest.approx(10.0)
