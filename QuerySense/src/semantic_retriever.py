"""Semantic retrieval with Sentence Transformers.

Encodes every searchable FAQ string into a dense embedding with
``sentence-transformers/all-MiniLM-L6-v2`` and ranks documents by cosine
similarity between the query embedding and the document embeddings.

Model loading is deliberately isolated in :func:`load_sentence_transformer`
so that a missing dependency or a failed download degrades gracefully into a
clear message instead of a crash.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from .base_retriever import BaseRetriever, DEFAULT_TOP_K, RetrievalHit  # noqa: F401

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MODEL_UNAVAILABLE_HINT = (
    "Semantic search could not be enabled. This usually means the "
    "sentence-transformers package is not installed, or the model weights could not "
    "be downloaded (no network access or a blocked Hugging Face endpoint). "
    "TF-IDF retrieval is unaffected and remains fully available."
)


class SemanticModelUnavailable(RuntimeError):
    """Raised when the sentence-transformer model cannot be loaded."""


def load_sentence_transformer(model_name: str = DEFAULT_MODEL_NAME, *, device: str = "cpu"):
    """Load a SentenceTransformer on CPU, converting any failure into a clear error.

    Kept as a module-level function so it can be wrapped in ``st.cache_resource``.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - depends on the local environment
        raise SemanticModelUnavailable(
            f"sentence-transformers is not importable ({type(exc).__name__}: {exc}). "
            + MODEL_UNAVAILABLE_HINT
        ) from exc

    try:
        return SentenceTransformer(model_name, device=device)
    except Exception as exc:  # pragma: no cover - depends on network access
        raise SemanticModelUnavailable(
            f"Could not load '{model_name}' ({type(exc).__name__}: {exc}). "
            + MODEL_UNAVAILABLE_HINT
        ) from exc


class SemanticRetriever(BaseRetriever):
    """Dense retrieval over normalized sentence embeddings."""

    name = "semantic"
    score_label = "cosine similarity (MiniLM embeddings)"

    def __init__(
        self,
        doc_ids: Sequence[str],
        texts: Sequence[str],
        *,
        model=None,
        model_name: str = DEFAULT_MODEL_NAME,
        batch_size: int = 32,
    ) -> None:
        super().__init__(doc_ids, texts)
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = model
        self.embeddings: np.ndarray | None = None

    def fit(self) -> "SemanticRetriever":
        if self.is_fitted:
            return self

        if self.model is None:
            load_start = time.perf_counter()
            self.model = load_sentence_transformer(self.model_name)
            self.model_load_seconds = time.perf_counter() - load_start

        start = time.perf_counter()
        # normalize_embeddings=True makes the dot product equal to cosine similarity.
        vectors = self.model.encode(
            self.texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        self.embeddings = np.asarray(vectors, dtype=np.float32)
        self.build_seconds = time.perf_counter() - start
        self.is_fitted = True
        return self

    def _score_all(self, query: str) -> list[float]:
        query_vector = self.model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        query_vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        return (self.embeddings @ query_vector).astype(float).tolist()

    @property
    def embedding_dim(self) -> int:
        if self.embeddings is None:
            return 0
        return int(self.embeddings.shape[1])

    def describe(self) -> dict:
        info = super().describe()
        info.update(
            {
                "model": self.model_name,
                "device": "cpu",
                "embedding_dim": self.embedding_dim,
                "normalized": True,
                "similarity": "cosine over L2-normalized MiniLM embeddings",
            }
        )
        return info


def semantic_is_available(model_name: str = DEFAULT_MODEL_NAME) -> tuple[bool, str]:
    """Non-fatal availability probe used by the UI before offering semantic mode."""
    try:
        load_sentence_transformer(model_name)
    except SemanticModelUnavailable as exc:
        return False, str(exc)
    except Exception as exc:  # pragma: no cover
        return False, f"Unexpected error while loading the model: {exc}"
    return True, ""


def build_semantic_retriever(
    doc_ids: Sequence[str],
    texts: Sequence[str],
    *,
    model=None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> SemanticRetriever:
    """Fit a :class:`SemanticRetriever`, surfacing a clear error if the model fails."""
    retriever = SemanticRetriever(doc_ids, texts, model=model, model_name=model_name)
    retriever.fit()
    return retriever
