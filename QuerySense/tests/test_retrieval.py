"""Tests for both retrieval strategies and their shared contract.

The semantic tests use the offline ``FakeSentenceTransformer`` fixture, so they
exercise the real retriever code path without downloading MiniLM.
"""

from __future__ import annotations

import pytest

from src.base_retriever import DEFAULT_TOP_K, MAX_TOP_K, MIN_TOP_K, clamp_top_k
from src.semantic_retriever import (
    SemanticModelUnavailable,
    SemanticRetriever,
    build_semantic_retriever,
    load_sentence_transformer,
)
from src.tfidf_retriever import TFIDFRetriever


@pytest.fixture
def tfidf(faqs) -> TFIDFRetriever:
    return TFIDFRetriever(faqs["id"].tolist(), faqs["search_text"].tolist()).fit()


@pytest.fixture
def semantic(faqs, fake_model) -> SemanticRetriever:
    return SemanticRetriever(
        faqs["id"].tolist(), faqs["search_text"].tolist(), model=fake_model
    ).fit()


class TestTopKClamping:
    @pytest.mark.parametrize(
        "supplied,expected",
        [(None, DEFAULT_TOP_K), (0, MIN_TOP_K), (-5, MIN_TOP_K), (1, 1), (10, 10), (99, MAX_TOP_K)],
    )
    def test_clamp(self, supplied, expected):
        assert clamp_top_k(supplied) == expected

    def test_non_numeric_falls_back_to_default(self):
        assert clamp_top_k("not-a-number") == DEFAULT_TOP_K


class TestTFIDFRetriever:
    def test_returns_exactly_k_results(self, tfidf):
        assert len(tfidf.search("how do I pay my fees", k=3)) == 3
        assert len(tfidf.search("how do I pay my fees", k=5)) == 5

    def test_ranks_are_sequential_from_one(self, tfidf):
        hits = tfidf.search("hostel maintenance problem", k=3)
        assert [hit.rank for hit in hits] == [1, 2, 3]

    def test_scores_are_cosine_similarities_in_range(self, tfidf):
        for hit in tfidf.search("library book loan period", k=5):
            assert 0.0 < hit.score <= 1.0 + 1e-6

    def test_scores_are_descending(self, tfidf):
        scores = [hit.score for hit in tfidf.search("attendance for exam eligibility", k=5)]
        assert scores == sorted(scores, reverse=True)

    def test_ranking_is_deterministic(self, tfidf):
        query = "how do I get a transcript"
        first = [hit.doc_id for hit in tfidf.search(query, k=5)]
        second = [hit.doc_id for hit in tfidf.search(query, k=5)]
        assert first == second

    def test_retrieves_the_expected_record(self, tfidf):
        hits = tfidf.search("What is the penalty for late fee payment?", k=3)
        assert hits[0].doc_id == "FEE-003"

    def test_empty_query_returns_nothing(self, tfidf):
        assert tfidf.search("") == []
        assert tfidf.search("   ") == []
        assert tfidf.search(None) == []

    def test_query_with_no_lexical_overlap_returns_nothing(self, tfidf):
        assert tfidf.search("zzzqqq xylophonic gibberish", k=3) == []

    def test_min_score_filters_weak_results(self, tfidf):
        query = "how do I pay my fees"
        unfiltered = tfidf.search(query, k=10)
        threshold = unfiltered[1].score + 1e-9
        filtered = tfidf.search(query, k=10, min_score=threshold)
        assert len(filtered) < len(unfiltered)
        assert all(hit.score >= threshold for hit in filtered)

    def test_document_ids_match_the_frame(self, tfidf, faqs):
        assert tfidf.doc_ids == faqs["id"].tolist()

    def test_fit_is_idempotent(self, tfidf):
        build_time = tfidf.build_seconds
        tfidf.fit()
        assert tfidf.build_seconds == build_time

    def test_search_before_fit_raises(self, faqs):
        unfitted = TFIDFRetriever(faqs["id"].tolist(), faqs["search_text"].tolist())
        with pytest.raises(RuntimeError, match="fitted"):
            unfitted.search("anything")

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            TFIDFRetriever(["a", "b"], ["only one"])

    def test_empty_collection_raises(self):
        with pytest.raises(ValueError, match="empty"):
            TFIDFRetriever([], [])

    def test_describe_exposes_build_metadata(self, tfidf):
        info = tfidf.describe()
        assert info["name"] == "tfidf"
        assert info["documents"] == 150
        assert info["vocabulary_size"] > 0


class TestSemanticRetriever:
    def test_returns_k_results_with_valid_scores(self, semantic):
        hits = semantic.search("library book loan period", k=3)
        assert len(hits) == 3
        for hit in hits:
            assert -1.0 <= hit.score <= 1.0 + 1e-6

    def test_ranking_is_deterministic(self, semantic):
        query = "how do I pay my fees"
        assert [h.doc_id for h in semantic.search(query, k=5)] == [
            h.doc_id for h in semantic.search(query, k=5)
        ]

    def test_empty_query_returns_nothing(self, semantic):
        assert semantic.search("") == []
        assert semantic.search("  ") == []

    def test_embeddings_are_normalized(self, semantic):
        import numpy as np

        norms = np.linalg.norm(semantic.embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4)

    def test_search_before_fit_raises(self, faqs, fake_model):
        unfitted = SemanticRetriever(
            faqs["id"].tolist(), faqs["search_text"].tolist(), model=fake_model
        )
        with pytest.raises(RuntimeError, match="fitted"):
            unfitted.search("anything")

    def test_describe_reports_the_model(self, semantic):
        info = semantic.describe()
        assert info["name"] == "semantic"
        assert "MiniLM" in info["model"]
        assert info["modality"] == "dense semantic"


class TestRetrieverInteroperability:
    def test_both_models_index_the_same_ids(self, tfidf, semantic):
        assert tfidf.doc_ids == semantic.doc_ids
        assert len(tfidf) == len(semantic)

    def test_hits_always_carry_a_valid_index(self, tfidf, semantic):
        for retriever in (tfidf, semantic):
            for hit in retriever.search("hostel wifi access", k=5):
                assert 0 <= hit.index < len(retriever)
                assert retriever.doc_ids[hit.index] == hit.doc_id

    def test_both_expose_the_shared_search_interface(self, tfidf, semantic, evaluation_queries):
        for retriever in (tfidf, semantic):
            assert isinstance(retriever.search_ids("refund policy", k=3), list)
            for query in evaluation_queries["query"].head(5):
                ids = retriever.search_ids(query, k=3)
                assert all(isinstance(doc_id, str) for doc_id in ids)


class TestModelFailureHandling:
    def test_build_raises_semantic_unavailable_when_loading_fails(self, faqs, monkeypatch):
        def boom(*args, **kwargs):
            raise SemanticModelUnavailable("simulated download failure")

        monkeypatch.setattr("src.semantic_retriever.load_sentence_transformer", boom)
        with pytest.raises(SemanticModelUnavailable, match="simulated download failure"):
            build_semantic_retriever(faqs["id"].tolist(), faqs["search_text"].tolist())

    def test_availability_probe_reports_failure_without_raising(self, monkeypatch):
        from src.semantic_retriever import semantic_is_available

        def boom(*args, **kwargs):
            raise SemanticModelUnavailable("no network")

        monkeypatch.setattr("src.semantic_retriever.load_sentence_transformer", boom)
        available, reason = semantic_is_available()
        assert available is False
        assert "no network" in reason

    def test_loader_wraps_import_errors(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("sentence_transformers"):
                raise ImportError("simulated missing package")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(SemanticModelUnavailable, match="not importable"):
            load_sentence_transformer("some/model")

    def test_a_working_model_still_loads(self, monkeypatch, fake_model):
        import src.semantic_retriever as semantic_module

        monkeypatch.setattr(semantic_module, "load_sentence_transformer", lambda *a, **k: fake_model)
        assert semantic_module.load_sentence_transformer() is fake_model
