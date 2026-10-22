"""Shared pytest fixtures and offline stand-ins for the embedding model.

The fake model keeps the test suite hermetic: no network access, no model
download and no torch dependency for the retrieval tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_evaluation_queries, load_faqs


class FakeSentenceTransformer:
    """Deterministic hashing bag-of-words encoder used in place of MiniLM."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.encoded_batches = 0

    def _embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in str(text).lower().split():
            vector[hash(token) % self.dim] += 1.0
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else vector

    def encode(self, texts, **kwargs):  # noqa: D401 - mirrors the real signature loosely
        if isinstance(texts, str):
            texts = [texts]
        self.encoded_batches += 1
        return np.vstack([self._embed_one(text) for text in texts])


@pytest.fixture(scope="session")
def faqs() -> pd.DataFrame:
    return load_faqs()


@pytest.fixture(scope="session")
def evaluation_queries() -> pd.DataFrame:
    return load_evaluation_queries()


@pytest.fixture
def fake_model() -> FakeSentenceTransformer:
    return FakeSentenceTransformer()


@pytest.fixture(scope="session")
def tiny_faqs() -> pd.DataFrame:
    """A five-record frame with known, hand-checkable retrieval behaviour."""
    from src.data_loader import validate_faqs

    return validate_faqs(
        pd.DataFrame(
            [
                {
                    "id": "T-1",
                    "category": "Admissions",
                    "question": "How do I apply for admission to the university?",
                    "answer": "Use the online admission portal and pay the fee.",
                    "keywords": "apply|admission|portal",
                    "source": "test",
                },
                {
                    "id": "T-2",
                    "category": "Examinations",
                    "question": "What attendance is required to write the exam?",
                    "answer": "Seventy-five percent attendance in each course.",
                    "keywords": "attendance|exam|eligible",
                    "source": "test",
                },
                {
                    "id": "T-3",
                    "category": "Library",
                    "question": "How long can I borrow a library book?",
                    "answer": "Two weeks, renewable once.",
                    "keywords": "library|borrow|loan period",
                    "source": "test",
                },
                {
                    "id": "T-4",
                    "category": "Hostel",
                    "question": "How do I report a hostel maintenance problem?",
                    "answer": "Raise a ticket with the warden.",
                    "keywords": "hostel|maintenance|complaint",
                    "source": "test",
                },
                {
                    "id": "T-5",
                    "category": "Fees",
                    "question": "How do I pay my semester fees online?",
                    "answer": "Pay through the fee portal by net banking or UPI.",
                    "keywords": "fees|payment|online",
                    "source": "test",
                },
            ]
        )
    )
