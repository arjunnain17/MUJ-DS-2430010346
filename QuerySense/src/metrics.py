"""Ranking metrics for information retrieval evaluation.

Definitions follow the capstone specification:

* ``Precision@k``  = relevant documents in the top k / k
  (the denominator is always k, even when fewer than k documents are returned,
  so a query with no matches scores 0 rather than being undefined)
* ``Recall@k``     = relevant documents in the top k / total relevant for the query
* ``MRR``          = mean of 1 / rank of the first relevant result, 0 when none is found

All helpers are pure functions over id sequences so they can be unit-tested
with hand-checked examples and never depend on a model prediction.
"""

from __future__ import annotations

from statistics import mean, median
from typing import Iterable, Sequence

import numpy as np


def _as_list(values: Iterable[str] | None) -> list[str]:
    if values is None:
        return []
    return [str(value) for value in values]


def relevant_retrieved(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int,
) -> list[str]:
    """Relevant ids that appear in the first ``k`` retrieved results (order preserved)."""
    if k <= 0:
        return []
    relevant = set(_as_list(relevant_ids))
    return [doc_id for doc_id in _as_list(retrieved_ids)[:k] if doc_id in relevant]


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Fraction of the top ``k`` slots that are relevant (denominator is k)."""
    if k <= 0:
        return 0.0
    return len(relevant_retrieved(retrieved_ids, relevant_ids, k)) / float(k)


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Fraction of the query's relevant documents found in the top ``k``."""
    relevant = set(_as_list(relevant_ids))
    if not relevant:
        return 0.0
    return len(relevant_retrieved(retrieved_ids, relevant, k)) / float(len(relevant))


def first_relevant_rank(retrieved_ids: Sequence[str], relevant_ids: Iterable[str]) -> int | None:
    """1-based rank of the first relevant result, or ``None`` when there is none."""
    relevant = set(_as_list(relevant_ids))
    for position, doc_id in enumerate(_as_list(retrieved_ids), start=1):
        if doc_id in relevant:
            return position
    return None


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    """1 / rank of the first relevant result, or 0.0 when nothing relevant is retrieved."""
    rank = first_relevant_rank(retrieved_ids, relevant_ids)
    return 0.0 if rank is None else 1.0 / rank


def hit_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> int:
    """1 when at least one relevant document is in the top ``k``, else 0."""
    return 1 if relevant_retrieved(retrieved_ids, relevant_ids, k) else 0


def average_precision(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Average precision over the top ``k`` (used for MAP if required)."""
    relevant = set(_as_list(relevant_ids))
    if not relevant or k <= 0:
        return 0.0
    hits = 0
    total = 0.0
    for position, doc_id in enumerate(_as_list(retrieved_ids)[:k], start=1):
        if doc_id in relevant:
            hits += 1
            total += hits / position
    return total / float(min(len(relevant), k))


def macro_average(values: Sequence[float]) -> float:
    """Unweighted mean, returning 0.0 for an empty sequence."""
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        return 0.0
    return float(mean(cleaned))


def latency_summary(latencies_ms: Sequence[float]) -> dict:
    """Mean and median query latency in milliseconds (per the specification)."""
    cleaned = [float(value) for value in latencies_ms if value is not None]
    if not cleaned:
        return {"mean_latency_ms": 0.0, "median_latency_ms": 0.0, "p95_latency_ms": 0.0, "queries": 0}
    return {
        "mean_latency_ms": float(mean(cleaned)),
        "median_latency_ms": float(median(cleaned)),
        "p95_latency_ms": float(np.percentile(cleaned, 95)),
        "queries": len(cleaned),
    }


def summarize_run(
    per_query: Sequence[dict],
    *,
    k: int,
    latencies_ms: Sequence[float] | None = None,
) -> dict:
    """Aggregate per-query metric rows into the headline evaluation numbers."""
    rows = list(per_query)
    precisions = [row["precision_at_k"] for row in rows]
    recalls = [row["recall_at_k"] for row in rows]
    rrs = [row["reciprocal_rank"] for row in rows]

    summary = {
        "evaluated_queries": len(rows),
        "k": k,
        "precision_at_k": macro_average(precisions),
        "recall_at_k": macro_average(recalls),
        "mrr": macro_average(rrs),
        "hit_rate_at_k": macro_average([row.get("hit_at_k", 0) for row in rows]),
        "map_at_k": macro_average([row.get("average_precision", 0.0) for row in rows]),
        "queries_with_no_relevant_retrieved": sum(1 for row in rows if row["reciprocal_rank"] == 0.0),
    }
    if latencies_ms is not None:
        summary.update(latency_summary(latencies_ms))
    return summary
