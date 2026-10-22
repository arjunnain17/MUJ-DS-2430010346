"""Loading and validating the knowledge base and evaluation query set.

All data access goes through this module so the Streamlit layer never touches
the filesystem directly, and so uploaded CSVs are validated with the same rules
as the bundled dataset.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .preprocessing import build_searchable_texts, normalize_keywords, normalize_whitespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_FAQ_PATH = DATA_DIR / "faqs.csv"
DEFAULT_EVAL_PATH = DATA_DIR / "evaluation_queries.csv"

FAQ_COLUMNS = ["id", "category", "question", "answer", "keywords", "source"]
REQUIRED_UPLOAD_COLUMNS = ["question", "answer"]
EVAL_COLUMNS = ["query_id", "query", "relevant_ids", "query_type"]

SAMPLE_DATA_NOTICE = (
    "The bundled knowledge base is synthetic, illustrative demo data created for this "
    "capstone project. It does not describe the official policies of any real institution."
)


class DataValidationError(ValueError):
    """Raised when a dataset is missing required columns or contains bad rows."""


def _read_csv(source: str | Path | io.IOBase) -> pd.DataFrame:
    """Read a CSV from a path or file-like object, failing with a clear message."""
    try:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise DataValidationError(f"File not found: {path}")
            return pd.read_csv(path, dtype=str, keep_default_na=False)
        return pd.read_csv(source, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError("The CSV file is empty.") from exc
    except pd.errors.ParserError as exc:
        raise DataValidationError(f"The CSV file could not be parsed: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise DataValidationError("The CSV file is not valid UTF-8 text.") from exc


def _check_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise DataValidationError(
            f"{label} is missing required column(s): {', '.join(missing)}. "
            f"Expected at least: {', '.join(required)}."
        )


def validate_faqs(df: pd.DataFrame, *, generate_missing_ids: bool = True) -> pd.DataFrame:
    """Validate and clean an FAQ frame.

    Rules enforced:
      * ``question`` and ``answer`` are required and must be non-empty.
      * ``id`` must be unique; missing IDs are generated when allowed.
      * optional columns are filled with safe defaults.
    """
    if df is None or df.empty:
        raise DataValidationError("The knowledge base contains no rows.")

    _check_columns(df, REQUIRED_UPLOAD_COLUMNS, "Knowledge base")

    cleaned = df.copy()
    cleaned.columns = [str(column).strip().lower() for column in cleaned.columns]

    for column in FAQ_COLUMNS:
        if column not in cleaned.columns:
            cleaned[column] = ""

    cleaned["question"] = cleaned["question"].map(normalize_whitespace)
    cleaned["answer"] = cleaned["answer"].map(normalize_whitespace)
    cleaned["category"] = cleaned["category"].map(normalize_whitespace)
    cleaned["keywords"] = cleaned["keywords"].map(normalize_whitespace)
    cleaned["source"] = cleaned["source"].map(normalize_whitespace)
    cleaned["id"] = cleaned["id"].map(normalize_whitespace)

    blank_questions = cleaned["question"] == ""
    if blank_questions.any():
        raise DataValidationError(
            f"{int(blank_questions.sum())} row(s) have an empty 'question' value."
        )

    blank_answers = cleaned["answer"] == ""
    if blank_answers.any():
        raise DataValidationError(
            f"{int(blank_answers.sum())} row(s) have an empty 'answer' value. "
            "Every FAQ record needs an answer to retrieve."
        )

    missing_id = cleaned["id"] == ""
    if missing_id.any():
        if not generate_missing_ids:
            raise DataValidationError(
                f"{int(missing_id.sum())} row(s) are missing an 'id' value."
            )
        for position in cleaned.index[missing_id]:
            cleaned.at[position, "id"] = f"FAQ-{position + 1:04d}"

    duplicates = cleaned["id"][cleaned["id"].duplicated()].unique().tolist()
    if duplicates:
        preview = ", ".join(str(value) for value in duplicates[:5])
        raise DataValidationError(
            f"Duplicate FAQ id(s) found: {preview}. Every record must have a unique id."
        )

    cleaned["category"] = cleaned["category"].replace("", "Uncategorized")
    cleaned["source"] = cleaned["source"].replace("", "user-upload")
    cleaned["keywords"] = cleaned["keywords"].map(
        lambda value: "|".join(normalize_keywords(value))
    )

    cleaned = cleaned.reset_index(drop=True)
    cleaned["search_text"] = build_searchable_texts(
        cleaned["question"], cleaned["keywords"], cleaned["category"]
    )
    return cleaned[FAQ_COLUMNS + ["search_text"]].reset_index(drop=True)


def load_faqs(path: str | Path = DEFAULT_FAQ_PATH) -> pd.DataFrame:
    """Load the bundled knowledge base from disk."""
    return validate_faqs(_read_csv(path))


def load_faqs_from_csv(content: bytes | str) -> pd.DataFrame:
    """Validate an uploaded CSV payload (bytes or text)."""
    if isinstance(content, bytes):
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DataValidationError("The uploaded file is not valid UTF-8 text.") from exc
    else:
        text = content

    if not text.strip():
        raise DataValidationError("The uploaded CSV file is empty.")

    return validate_faqs(_read_csv(io.StringIO(text)))


def load_evaluation_queries(path: str | Path = DEFAULT_EVAL_PATH) -> pd.DataFrame:
    """Load the labeled evaluation queries and parse the relevance lists."""
    df = _read_csv(path)
    _check_columns(df, ["query", "relevant_ids"], "Evaluation set")

    if df.empty:
        raise DataValidationError("The evaluation query set contains no rows.")

    cleaned = df.copy()
    cleaned.columns = [str(column).strip().lower() for column in cleaned.columns]
    if "query_id" not in cleaned.columns:
        cleaned["query_id"] = [f"Q{index + 1:02d}" for index in range(len(cleaned))]
    if "query_type" not in cleaned.columns:
        cleaned["query_type"] = "unspecified"

    cleaned["query"] = cleaned["query"].map(normalize_whitespace)
    cleaned["query_id"] = cleaned["query_id"].map(normalize_whitespace)
    cleaned["query_type"] = cleaned["query_type"].map(normalize_whitespace)

    cleaned = cleaned[cleaned["query"] != ""].reset_index(drop=True)

    relevance_lists: list[list[str]] = []
    for position, raw in enumerate(cleaned["relevant_ids"]):
        ids = [
            normalize_whitespace(part)
            for part in str(raw).replace(";", "|").replace(",", "|").split("|")
        ]
        ids = [value for value in ids if value]
        if not ids:
            raise DataValidationError(
                f"Evaluation query '{cleaned.at[position, 'query_id']}' has no relevant ids."
            )
        relevance_lists.append(ids)

    cleaned["relevant_id_list"] = relevance_lists
    cleaned["relevant_ids"] = ["|".join(ids) for ids in relevance_lists]
    return cleaned[EVAL_COLUMNS + ["relevant_id_list"]].reset_index(drop=True)


def validate_evaluation_against_faqs(
    queries: pd.DataFrame, faqs: pd.DataFrame
) -> list[str]:
    """Return warnings for ground-truth ids that do not exist in the knowledge base."""
    known = set(faqs["id"])
    warnings: list[str] = []
    for row in queries.itertuples(index=False):
        unknown = [faq_id for faq_id in row.relevant_id_list if faq_id not in known]
        if unknown:
            warnings.append(
                f"{row.query_id}: relevant id(s) not present in the knowledge base: "
                f"{', '.join(unknown)}"
            )
    return warnings


def category_distribution(faqs: pd.DataFrame) -> pd.DataFrame:
    """Count records per category, largest first."""
    counts = (
        faqs["category"]
        .value_counts()
        .rename_axis("category")
        .reset_index(name="records")
    )
    return counts.sort_values(["records", "category"], ascending=[False, True]).reset_index(
        drop=True
    )


def filter_by_category(faqs: pd.DataFrame, category: str | None) -> pd.DataFrame:
    """Return all rows, or only those in ``category`` when one is given."""
    if not category or category == "All categories":
        return faqs
    return faqs[faqs["category"] == category]


def knowledge_base_fingerprint(faqs: pd.DataFrame) -> str:
    """Stable hash of FAQ ids and questions, used to decide when to rebuild indexes."""
    import hashlib

    hasher = hashlib.sha256()
    for faq_id, question in zip(faqs["id"], faqs["question"]):
        hasher.update(f"{faq_id}\u241f{question}\u241e".encode("utf-8"))
    return hasher.hexdigest()[:16]


def records_as_dicts(faqs: pd.DataFrame) -> list[dict]:
    """Convert the frame into plain dicts for display helpers."""
    return faqs.to_dict(orient="records")


def combine_datasets(*frames: pd.DataFrame) -> pd.DataFrame:
    """Concatenate FAQ frames, re-validating the merged result."""
    non_empty: Sequence[pd.DataFrame] = [frame for frame in frames if frame is not None and not frame.empty]
    if not non_empty:
        raise DataValidationError("No knowledge base records were provided.")
    return validate_faqs(pd.concat(non_empty, ignore_index=True), generate_missing_ids=False)
