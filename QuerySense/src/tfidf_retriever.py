"""Baseline retrieval: TF-IDF vectors ranked by cosine similarity.

This is the lexical baseline. It matches on shared surface forms, so it is
strong when the user reuses the words in the knowledge base and weak when the
user paraphrases with different vocabulary.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .base_retriever import BaseRetriever, DEFAULT_TOP_K, RetrievalHit  # noqa: F401


class TFIDFRetriever(BaseRetriever):
    """Sparse bag-of-words retrieval with L2-normalized TF-IDF vectors."""

    name = "tfidf"
    score_label = "cosine similarity (TF-IDF)"

    def __init__(
        self,
        doc_ids: Sequence[str],
        texts: Sequence[str],
        *,
        ngram_range: tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
        min_df: int = 1,
        max_features: int | None = 20000,
        stop_words: str | None = "english",
    ) -> None:
        super().__init__(doc_ids, texts)
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.min_df = min_df
        self.max_features = max_features
        self.stop_words = stop_words

        self.vectorizer: TfidfVectorizer | None = None
        self.doc_matrix = None

    def fit(self) -> "TFIDFRetriever":
        if self.is_fitted:
            return self

        start = time.perf_counter()
        safe_min_df = min(self.min_df, max(1, len(self.texts)))
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=self.stop_words,
            ngram_range=self.ngram_range,
            sublinear_tf=self.sublinear_tf,
            min_df=safe_min_df,
            max_features=self.max_features,
            strip_accents="unicode",
        )
        # sklearn L2-normalizes rows by default, so the dot product is cosine similarity.
        self.doc_matrix = self.vectorizer.fit_transform(self.texts)
        self.build_seconds = time.perf_counter() - start
        self.is_fitted = True
        return self

    def _score_all(self, query: str) -> list[float]:
        query_vector = self.vectorizer.transform([query])
        sparse_scores = self.doc_matrix.dot(query_vector.T)
        return np.asarray(sparse_scores.toarray()).ravel().astype(float).tolist()

    @property
    def vocabulary_size(self) -> int:
        if self.vectorizer is None or not hasattr(self.vectorizer, "vocabulary_"):
            return 0
        return len(self.vectorizer.vocabulary_)

    def top_terms(self, query: str, limit: int = 8) -> list[str]:
        """The query's highest-weighted TF-IDF terms, useful for explaining misses."""
        if not self.is_fitted:
            return []
        query_vector = self.vectorizer.transform([query])
        row = np.asarray(query_vector.toarray()).ravel()
        if row.size == 0:
            return []
        names = self.vectorizer.get_feature_names_out()
        order = np.argsort(-row)[:limit]
        return [str(names[index]) for index in order if row[index] > 0]

    def describe(self) -> dict:
        info = super().describe()
        info.update(
            {
                "vectorizer": "sklearn TfidfVectorizer",
                "ngram_range": list(self.ngram_range),
                "sublinear_tf": self.sublinear_tf,
                "stop_words": self.stop_words or "none",
                "vocabulary_size": self.vocabulary_size,
                "similarity": "cosine over L2-normalized TF-IDF vectors",
            }
        )
        return info
