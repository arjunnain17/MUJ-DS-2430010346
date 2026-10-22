"""Shared retriever contract so TF-IDF and semantic retrieval are interchangeable."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

MIN_TOP_K = 1
MAX_TOP_K = 10
DEFAULT_TOP_K = 3


@dataclass(frozen=True)
class RetrievalHit:
    """A single ranked result.

    ``score`` is a model-specific similarity score. Scores from different
    retrievers are NOT calibrated and must not be compared across models or
    read as probabilities.
    """

    rank: int
    doc_id: str
    score: float
    index: int

    def as_dict(self) -> dict:
        return {
            "rank": self.rank,
            "id": self.doc_id,
            "score": self.score,
            "index": self.index,
        }


def clamp_top_k(k: int | None) -> int:
    """Keep ``k`` inside the supported 1..10 range."""
    if k is None:
        return DEFAULT_TOP_K
    try:
        value = int(k)
    except (TypeError, ValueError):
        return DEFAULT_TOP_K
    return max(MIN_TOP_K, min(MAX_TOP_K, value))


class BaseRetriever(ABC):
    """Common interface implemented by every retrieval strategy.

    Subclasses hold an ordered list of document ids plus the searchable text
    that was indexed, so evaluation can compare both models on identical ids.
    """

    name: str = "base"
    score_label: str = "similarity"

    def __init__(self, doc_ids: Sequence[str], texts: Sequence[str]) -> None:
        if len(doc_ids) != len(texts):
            raise ValueError(
                f"doc_ids and texts must be the same length (got {len(doc_ids)} and {len(texts)})."
            )
        if not doc_ids:
            raise ValueError("Cannot build an index over an empty document collection.")

        self.doc_ids: list[str] = [str(value) for value in doc_ids]
        self.texts: list[str] = ["" if value is None else str(value) for value in texts]
        # Model loading and index construction are timed separately so that
        # steady-state query latency is never conflated with setup cost.
        self.model_load_seconds: float = 0.0
        self.build_seconds: float = 0.0
        self.is_fitted: bool = False

    def __len__(self) -> int:
        return len(self.doc_ids)

    @abstractmethod
    def fit(self) -> "BaseRetriever":
        """Build the index. Must be idempotent."""

    @abstractmethod
    def _score_all(self, query: str) -> list[float]:
        """Return a similarity score for every indexed document."""

    def search(
        self,
        query: str | None,
        k: int = DEFAULT_TOP_K,
        min_score: float | None = None,
    ) -> list[RetrievalHit]:
        """Rank documents for ``query`` and return the top ``k`` hits.

        Empty or whitespace-only queries return an empty list. Non-positive
        scores are dropped, so a query that shares no vocabulary with the
        knowledge base yields no hits rather than misleading ones.
        """
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} retriever must be fitted before searching.")

        from .preprocessing import normalize_query

        normalized = normalize_query(query)
        if not normalized:
            return []

        scores = self._score_all(normalized)
        if not scores:
            return []

        limit = clamp_top_k(k)

        ranked = sorted(
            ((index, score) for index, score in enumerate(scores) if score > 0.0),
            key=lambda pair: (-pair[1], pair[0]),
        )

        if min_score is not None:
            ranked = [pair for pair in ranked if pair[1] >= float(min_score)]

        return [
            RetrievalHit(rank=position, doc_id=self.doc_ids[index], score=float(score), index=index)
            for position, (index, score) in enumerate(ranked[:limit], start=1)
        ]

    def search_ids(
        self,
        query: str | None,
        k: int = DEFAULT_TOP_K,
        min_score: float | None = None,
    ) -> list[str]:
        """Convenience wrapper returning only the ranked document ids."""
        return [hit.doc_id for hit in self.search(query, k=k, min_score=min_score)]

    def describe(self) -> dict:
        """Metadata shown in the UI and written to results/metrics.json."""
        return {
            "name": self.name,
            "score_label": self.score_label,
            "documents": len(self),
            "model_load_seconds": round(self.model_load_seconds, 4),
            "build_seconds": round(self.build_seconds, 4),
            "modality": "sparse lexical" if self.name == "tfidf" else "dense semantic",
        }
