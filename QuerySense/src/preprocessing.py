"""Text normalization and searchable-text construction.

Preprocessing is deliberately light. Aggressive stemming or stop-word removal
hurts the semantic model more than it helps, so we only normalize whitespace,
handle missing values, and build one searchable string per record.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

_WHITESPACE_RE = re.compile(r"\s+")
_KEYWORD_SPLIT_RE = re.compile(r"[|;,/]+")

MISSING_TOKENS = {"", "nan", "none", "null", "na", "n/a", "-"}


def normalize_whitespace(value: object) -> str:
    """Collapse runs of whitespace and trim the result.

    Missing values (``None``, ``NaN``, ``"nan"``, ``"null"``, ...) become an
    empty string so downstream code never has to special-case them.
    """
    if value is None:
        return ""
    text = str(value)
    if text.strip().lower() in MISSING_TOKENS:
        return ""
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_keywords(value: object) -> list[str]:
    """Split a keyword field into a clean list of individual keywords."""
    text = normalize_whitespace(value)
    if not text:
        return []
    parts = _KEYWORD_SPLIT_RE.split(text)
    seen: set[str] = set()
    keywords: list[str] = []
    for part in parts:
        token = normalize_whitespace(part).lower()
        if token and token not in seen:
            seen.add(token)
            keywords.append(token)
    return keywords


def build_searchable_text(
    question: object,
    keywords: object = "",
    category: object = "",
    *,
    include_category: bool = True,
) -> str:
    """Build the string that gets indexed by both retrievers.

    The question is the primary signal; keywords add vocabulary that users
    actually type, and the category adds coarse topical context. Answers are
    intentionally excluded so the retrievers rank *questions*, not long answers
    that would otherwise dominate the similarity scores.
    """
    question_text = normalize_whitespace(question)
    keyword_text = " ".join(normalize_keywords(keywords))
    category_text = normalize_whitespace(category) if include_category else ""

    segments = [segment for segment in (question_text, keyword_text, category_text) if segment]
    return " ".join(segments)


def build_searchable_texts(
    questions: Sequence[object],
    keywords: Sequence[object] | None = None,
    categories: Sequence[object] | None = None,
    *,
    include_category: bool = True,
) -> list[str]:
    """Vectorized wrapper around :func:`build_searchable_text`."""
    if keywords is None:
        keywords = [""] * len(questions)
    if categories is None:
        categories = [""] * len(questions)

    return [
        build_searchable_text(q, k, c, include_category=include_category)
        for q, k, c in zip(questions, keywords, categories)
    ]


def normalize_query(query: object) -> str:
    """Normalize a user-supplied query, returning "" for empty input."""
    return normalize_whitespace(query)


def is_blank_query(query: object) -> bool:
    """True when the query carries no searchable characters."""
    text = normalize_whitespace(query)
    return not any(ch.isalnum() for ch in text)


def truncate(text: object, max_chars: int = 240, suffix: str = "...") -> str:
    """Shorten a string for compact table display."""
    normalized = normalize_whitespace(text)
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - len(suffix)].rstrip() + suffix


def dedupe_preserving_order(values: Iterable[str]) -> list[str]:
    """De-duplicate while keeping the original ordering."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
