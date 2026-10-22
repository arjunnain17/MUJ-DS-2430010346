"""Tests for CSV loading, schema validation and identifier integrity."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from src.data_loader import (
    DataValidationError,
    category_distribution,
    combine_datasets,
    filter_by_category,
    knowledge_base_fingerprint,
    load_evaluation_queries,
    load_faqs,
    load_faqs_from_csv,
    validate_evaluation_against_faqs,
    validate_faqs,
)


class TestBundledDataset:
    def test_loads_with_expected_shape(self, faqs):
        assert len(faqs) == 150
        assert list(faqs.columns) == [
            "id",
            "category",
            "question",
            "answer",
            "keywords",
            "source",
            "search_text",
        ]

    def test_ids_are_unique_and_non_empty(self, faqs):
        assert faqs["id"].is_unique
        assert (faqs["id"].str.strip() != "").all()

    def test_required_fields_are_populated(self, faqs):
        for column in ("question", "answer", "category", "source", "search_text"):
            assert (faqs[column].str.strip() != "").all(), f"{column} has empty values"

    def test_categories_are_reasonable(self, faqs):
        assert faqs["category"].nunique() >= 6

    def test_search_text_contains_the_question(self, faqs):
        row = faqs.iloc[0]
        assert row["question"][:20] in row["search_text"]

    def test_category_distribution_totals_all_records(self, faqs):
        distribution = category_distribution(faqs)
        assert distribution["records"].sum() == len(faqs)
        # Sorted descending by record count.
        assert distribution["records"].is_monotonic_decreasing

    def test_filter_by_category(self, faqs):
        category = faqs["category"].iloc[0]
        subset = filter_by_category(faqs, category)
        assert len(subset) == int((faqs["category"] == category).sum())
        assert len(filter_by_category(faqs, "All categories")) == len(faqs)
        assert len(filter_by_category(faqs, None)) == len(faqs)

    def test_fingerprint_is_stable_and_content_sensitive(self, faqs):
        assert knowledge_base_fingerprint(faqs) == knowledge_base_fingerprint(faqs)
        tampered = faqs.copy()
        tampered.loc[0, "question"] = "Completely different question?"
        assert knowledge_base_fingerprint(tampered) != knowledge_base_fingerprint(faqs)


class TestEvaluationSet:
    def test_has_at_least_thirty_queries(self, evaluation_queries):
        assert len(evaluation_queries) >= 30

    def test_relevance_lists_are_parsed(self, evaluation_queries):
        assert evaluation_queries["relevant_id_list"].map(lambda ids: len(ids) > 0).all()
        assert evaluation_queries["query_id"].is_unique

    def test_ground_truth_ids_exist_in_the_knowledge_base(self, evaluation_queries, faqs):
        assert validate_evaluation_against_faqs(evaluation_queries, faqs) == []

    def test_unknown_ground_truth_id_is_reported(self, evaluation_queries, faqs):
        broken = evaluation_queries.copy()
        broken.loc[0, "relevant_id_list"] = ["DOES-NOT-EXIST"]
        warnings = validate_evaluation_against_faqs(broken, faqs)
        assert len(warnings) == 1
        assert "DOES-NOT-EXIST" in warnings[0]

    def test_queries_cover_multiple_types(self, evaluation_queries):
        assert evaluation_queries["query_type"].nunique() >= 4

    def test_no_query_is_identical_to_a_faq_question(self, evaluation_queries, faqs):
        """Evaluation queries must not simply repeat a FAQ's exact wording."""
        faq_questions = set(faqs["question"].str.lower().str.strip())
        duplicates = [
            query
            for query in evaluation_queries["query"].str.lower().str.strip()
            if query in faq_questions
        ]
        assert duplicates == []

    def test_missing_relevant_ids_column_raises(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("query_id,query\nQ1,hello\n", encoding="utf-8")
        with pytest.raises(DataValidationError, match="relevant_ids"):
            load_evaluation_queries(path)


class TestFaqValidation:
    def test_missing_required_columns_raises(self):
        with pytest.raises(DataValidationError) as exc:
            validate_faqs(pd.DataFrame([{"id": "A", "question": "Q"}]))
        assert "answer" in str(exc.value)

    def test_empty_frame_raises(self):
        with pytest.raises(DataValidationError):
            validate_faqs(pd.DataFrame(columns=["question", "answer"]))

    def test_empty_question_raises(self):
        frame = pd.DataFrame([{"question": "   ", "answer": "a"}])
        with pytest.raises(DataValidationError, match="empty 'question'"):
            validate_faqs(frame)

    def test_empty_answer_raises(self):
        frame = pd.DataFrame([{"question": "why", "answer": ""}])
        with pytest.raises(DataValidationError, match="empty 'answer'"):
            validate_faqs(frame)

    def test_duplicate_ids_raise(self):
        frame = pd.DataFrame(
            [
                {"id": "X", "question": "q1", "answer": "a1"},
                {"id": "X", "question": "q2", "answer": "a2"},
            ]
        )
        with pytest.raises(DataValidationError, match="Duplicate"):
            validate_faqs(frame)

    def test_missing_ids_are_generated_when_allowed(self):
        frame = pd.DataFrame(
            [
                {"question": "q1", "answer": "a1"},
                {"question": "q2", "answer": "a2"},
            ]
        )
        cleaned = validate_faqs(frame, generate_missing_ids=True)
        assert len(cleaned["id"].unique()) == 2
        assert (cleaned["id"].str.startswith("FAQ-")).all()

    def test_missing_ids_raise_when_not_allowed(self):
        frame = pd.DataFrame([{"question": "q1", "answer": "a1"}])
        with pytest.raises(DataValidationError, match="missing an 'id'"):
            validate_faqs(frame, generate_missing_ids=False)

    def test_optional_columns_get_safe_defaults(self):
        cleaned = validate_faqs(pd.DataFrame([{"question": "q", "answer": "a"}]))
        assert cleaned.loc[0, "category"] == "Uncategorized"
        assert cleaned.loc[0, "source"] == "user-upload"
        assert cleaned.loc[0, "search_text"].strip() != ""

    def test_column_names_are_case_insensitive(self):
        cleaned = validate_faqs(pd.DataFrame([{"Question": "q", "Answer": "a"}]))
        assert cleaned.loc[0, "question"] == "q"

    def test_whitespace_is_normalized(self):
        cleaned = validate_faqs(
            pd.DataFrame([{"question": "  why   so \n many  spaces ", "answer": " a  b "}])
        )
        assert cleaned.loc[0, "question"] == "why so many spaces"
        assert cleaned.loc[0, "answer"] == "a b"


class TestCsvUpload:
    def test_valid_upload_is_accepted(self):
        payload = "question,answer,category\nHow do I apply?,Use the portal.,Admissions\n"
        cleaned = load_faqs_from_csv(payload.encode("utf-8"))
        assert len(cleaned) == 1
        assert cleaned.loc[0, "category"] == "Admissions"

    def test_upload_missing_answer_column_is_rejected(self):
        with pytest.raises(DataValidationError, match="answer"):
            load_faqs_from_csv("question,category\nHow do I apply?,Admissions\n")

    def test_upload_missing_question_column_is_rejected(self):
        with pytest.raises(DataValidationError, match="question"):
            load_faqs_from_csv("answer,category\nUse the portal.,Admissions\n")

    def test_empty_upload_is_rejected(self):
        with pytest.raises(DataValidationError):
            load_faqs_from_csv("")

    def test_upload_with_duplicate_ids_is_rejected(self):
        payload = "id,question,answer\nD1,a,b\nD1,c,d\n"
        with pytest.raises(DataValidationError, match="Duplicate"):
            load_faqs_from_csv(payload)

    def test_upload_with_bom_is_handled(self):
        payload = "\ufeffquestion,answer\nHow?,Like this.\n"
        assert len(load_faqs_from_csv(payload.encode("utf-8"))) == 1

    def test_upload_does_not_touch_the_bundled_dataset(self, faqs):
        """Uploading must never mutate or overwrite the sample data on disk."""
        before = load_faqs()
        load_faqs_from_csv("question,answer\nExtra?,Yes.\n")
        after = load_faqs()
        pd.testing.assert_frame_equal(before, after)
        assert len(after) == len(faqs)


class TestDatasetCombination:
    def test_combine_keeps_unique_ids(self, tiny_faqs):
        extra = tiny_faqs.iloc[:2].copy()
        extra["id"] = ["EXTRA-1", "EXTRA-2"]
        combined = combine_datasets(tiny_faqs, extra)
        assert len(combined) == len(tiny_faqs) + 2
        assert combined["id"].is_unique

    def test_combine_rejects_conflicting_ids(self, tiny_faqs):
        clash = tiny_faqs.copy()
        clash["question"] = "A different question entirely?"
        with pytest.raises(DataValidationError, match="Duplicate"):
            combine_datasets(tiny_faqs, clash)

    def test_combine_with_nothing_raises(self):
        with pytest.raises(DataValidationError):
            combine_datasets()


def test_read_csv_reports_missing_file(tmp_path):
    with pytest.raises(DataValidationError, match="File not found"):
        load_faqs(tmp_path / "nope.csv")


def test_read_csv_reports_malformed_content(tmp_path):
    path = tmp_path / "broken.csv"
    path.write_text("question,answer\n\"unterminated,value\n", encoding="utf-8")
    with pytest.raises(DataValidationError):
        load_faqs(path)


def test_bytes_upload_uses_utf8():
    with pytest.raises(DataValidationError, match="UTF-8"):
        load_faqs_from_csv(b"\xff\xfe\x00\x00not-utf8")
