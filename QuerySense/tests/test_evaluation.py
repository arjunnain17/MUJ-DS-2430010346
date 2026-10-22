"""Tests for the evaluation pipeline.

These verify that reported metrics come from the retriever's own predictions
(using a scripted retriever with hand-computed expected values), that both
models see identical queries, and that a missing semantic model is reported
rather than silently replaced with invented numbers.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.base_retriever import BaseRetriever
from src.evaluation import (
    build_metrics_payload,
    comparison_table,
    evaluate_all,
    evaluate_retriever,
    qualitative_examples,
    save_results,
)
from src.semantic_retriever import SemanticRetriever
from src.tfidf_retriever import TFIDFRetriever


class ScriptedRetriever(BaseRetriever):
    """Returns pre-seeded scores so expected metrics can be computed by hand."""

    name = "scripted"

    def __init__(self, doc_ids, texts, scores_for_query):
        super().__init__(doc_ids, texts)
        self.scores_for_query = scores_for_query

    def fit(self):
        self.is_fitted = True
        return self

    def _score_all(self, query):
        return self.scores_for_query[query]


@pytest.fixture
def scripted_queries() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "query_id": "S1",
                "query": "q1",
                "query_type": "test",
                "relevant_ids": "A",
                "relevant_id_list": ["A"],
            },
            {
                "query_id": "S2",
                "query": "q2",
                "query_type": "test",
                "relevant_ids": "C",
                "relevant_id_list": ["C"],
            },
            {
                "query_id": "S3",
                "query": "q3",
                "query_type": "test",
                "relevant_ids": "Z",
                "relevant_id_list": ["Z"],
            },
        ]
    )


@pytest.fixture
def scripted_retriever() -> ScriptedRetriever:
    return ScriptedRetriever(
        doc_ids=["A", "B", "C"],
        texts=["alpha", "beta", "gamma"],
        scores_for_query={
            # q1 -> A, B, C  (relevant A at rank 1)
            "q1": [0.9, 0.5, 0.1],
            # q2 -> B, A, C  (relevant C at rank 3)
            "q2": [0.5, 0.9, 0.1],
            # q3 -> none of the retrieved ids are relevant
            "q3": [0.9, 0.5, 0.4],
        },
    ).fit()


class TestEvaluateRetriever:
    def test_per_query_rows_match_the_query_set(self, scripted_retriever, scripted_queries):
        outcome = evaluate_retriever(scripted_retriever, scripted_queries, k=3)
        assert len(outcome["per_query"]) == len(scripted_queries)
        assert outcome["summary"]["evaluated_queries"] == 3

    def test_metrics_match_hand_computed_values(self, scripted_retriever, scripted_queries):
        summary = evaluate_retriever(scripted_retriever, scripted_queries, k=3)["summary"]
        # P@3: (1/3 + 1/3 + 0) / 3
        assert summary["precision_at_k"] == pytest.approx(2 / 9)
        # R@3: (1 + 1 + 0) / 3
        assert summary["recall_at_k"] == pytest.approx(2 / 3)
        # MRR: (1 + 1/3 + 0) / 3
        assert summary["mrr"] == pytest.approx(4 / 9)
        assert summary["queries_with_no_relevant_retrieved"] == 1

    def test_per_query_details_are_recorded(self, scripted_retriever, scripted_queries):
        per_query = evaluate_retriever(scripted_retriever, scripted_queries, k=3)["per_query"]
        row = per_query[per_query["query_id"] == "S2"].iloc[0]
        assert row["retrieved_ids"] == "B|A|C"
        assert row["first_relevant_rank"] == 3
        assert row["reciprocal_rank"] == pytest.approx(1 / 3)
        assert row["correct"] in (True, 1)

    def test_missed_relevant_ids_are_reported(self, scripted_retriever, scripted_queries):
        per_query = evaluate_retriever(scripted_retriever, scripted_queries, k=3)["per_query"]
        row = per_query[per_query["query_id"] == "S3"].iloc[0]
        assert row["missed_relevant_ids"] == "Z"
        assert row["first_relevant_rank"] == 0

    def test_latency_is_measured_per_query(self, scripted_retriever, scripted_queries):
        outcome = evaluate_retriever(scripted_retriever, scripted_queries, k=3)
        assert (outcome["per_query"]["latency_ms"] >= 0).all()
        assert outcome["summary"]["median_latency_ms"] >= 0
        assert outcome["summary"]["queries"] == 3

    def test_empty_query_set_raises(self, scripted_retriever):
        with pytest.raises(ValueError, match="empty query set"):
            evaluate_retriever(scripted_retriever, pd.DataFrame())

    def test_k_is_clamped_to_supported_range(self, scripted_retriever, scripted_queries):
        summary = evaluate_retriever(scripted_retriever, scripted_queries, k=99)["summary"]
        assert summary["k"] == 10


class TestEvaluateAll:
    def test_both_models_evaluate_the_same_queries(self, faqs, tiny_faqs, fake_model):
        from src.data_loader import load_evaluation_queries

        queries = load_evaluation_queries()
        id_lists = [faqs["id"].tolist(), faqs["id"].tolist()]
        retrievers = {
            "tfidf": TFIDFRetriever(*id_lists).fit(),
            "semantic": SemanticRetriever(*id_lists, model=fake_model).fit(),
        }
        results = evaluate_all(retrievers, queries, k=3)

        assert set(results) == {"tfidf", "semantic"}
        for result in results.values():
            assert len(result["per_query"]) == len(queries)
        tfidf_queries = results["tfidf"]["per_query"]["query_id"].tolist()
        semantic_queries = results["semantic"]["per_query"]["query_id"].tolist()
        assert tfidf_queries == semantic_queries

    def test_no_retrievers_raises(self, evaluation_queries):
        with pytest.raises(ValueError, match="No retrievers"):
            evaluate_all({}, evaluation_queries)


class TestComparisonTable:
    def test_one_row_per_model_with_expected_columns(self, scripted_retriever, scripted_queries):
        results = {"scripted": evaluate_retriever(scripted_retriever, scripted_queries, k=3)}
        table = comparison_table(results, k=3)
        assert len(table) == 1
        for column in ("Model", "Precision@3", "Recall@3", "MRR", "Median latency (ms)"):
            assert column in table.columns


class TestMetricsPayload:
    def test_payload_reports_unavailable_models_without_inventing_numbers(
        self, scripted_retriever, scripted_queries, faqs
    ):
        results = {"scripted": evaluate_retriever(scripted_retriever, scripted_queries, k=3)}
        payload = build_metrics_payload(
            results, faqs, scripted_queries, k=3, unavailable={"semantic": "model download blocked"}
        )
        assert payload["models"]["scripted"]["available"] is True
        assert payload["models"]["semantic"]["available"] is False
        assert payload["models"]["semantic"]["reason"] == "model download blocked"
        assert "metrics" not in payload["models"]["semantic"]

    def test_payload_records_dataset_provenance(
        self, scripted_retriever, scripted_queries, faqs
    ):
        results = {"scripted": evaluate_retriever(scripted_retriever, scripted_queries, k=3)}
        payload = build_metrics_payload(results, faqs, scripted_queries, k=3)
        assert payload["knowledge_base"]["records"] == len(faqs)
        assert payload["evaluation_set"]["queries"] == 3
        assert payload["environment"]["python"]


class TestSaveResults:
    def test_writes_three_artifacts(
        self, scripted_retriever, scripted_queries, faqs, tmp_path
    ):
        results = {"scripted": evaluate_retriever(scripted_retriever, scripted_queries, k=3)}
        paths = save_results(results, faqs, scripted_queries, k=3, output_dir=tmp_path)

        for path in paths.values():
            assert path.exists()

        payload = json.loads(paths["metrics"].read_text(encoding="utf-8"))
        assert payload["models"]["scripted"]["metrics"]["evaluated_queries"] == 3

        results_frame = pd.read_csv(paths["results"])
        assert list(results_frame["model"].unique()) == ["scripted"]
        assert len(results_frame) == 3


class TestQualitativeExamples:
    def test_requires_both_models(self, scripted_retriever, scripted_queries, faqs):
        results = {"tfidf": evaluate_retriever(scripted_retriever, scripted_queries, k=3)}
        assert qualitative_examples(results, faqs, k=3).empty

    def test_classifies_agreement_divergence_and_failure(self, faqs, evaluation_queries, fake_model):
        doc_ids = faqs["id"].tolist()
        texts = faqs["search_text"].tolist()
        retrievers = {
            "tfidf": TFIDFRetriever(doc_ids, texts).fit(),
            "semantic": SemanticRetriever(doc_ids, texts, model=fake_model).fit(),
        }
        results = evaluate_all(retrievers, evaluation_queries, k=3)
        table = qualitative_examples(results, faqs, k=3)

        assert len(table) >= 5
        assert set(table["case_type"]).issubset(
            {
                "both_succeed_same_rank",
                "both_succeed_rank_differs",
                "semantic_only",
                "tfidf_only",
                "both_fail",
            }
        )
        for column in ("query", "tfidf_top_ids", "semantic_top_ids", "case_label"):
            assert column in table.columns


class TestFullPipelineIntegrity:
    def test_metrics_are_not_hardcoded(self, faqs, fake_model, tmp_path):
        """Re-running with a different model must change the reported numbers."""
        from src.data_loader import load_evaluation_queries

        queries = load_evaluation_queries()
        doc_ids = faqs["id"].tolist()
        texts = faqs["search_text"].tolist()

        realish = evaluate_all(
            {"tfidf": TFIDFRetriever(doc_ids, texts).fit()}, queries, k=3
        )
        degraded = evaluate_all(
            {"tfidf": TFIDFRetriever(doc_ids, ["constant text"] * len(texts)).fit()},
            queries,
            k=3,
        )
        assert realish["tfidf"]["summary"]["mrr"] > degraded["tfidf"]["summary"]["mrr"]
