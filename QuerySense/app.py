"""QuerySense — Intelligent Semantic Search & FAQ Retrieval System.

Streamlit front end. All retrieval, evaluation and data-access logic lives in
``src/``; this module is responsible only for presentation and session state.
"""

from __future__ import annotations

import html
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.base_retriever import DEFAULT_TOP_K, MAX_TOP_K, MIN_TOP_K, RetrievalHit
from src.data_loader import (
    SAMPLE_DATA_NOTICE,
    DataValidationError,
    category_distribution,
    knowledge_base_fingerprint,
    load_evaluation_queries,
    load_faqs,
    load_faqs_from_csv,
)
from src.evaluation import comparison_table, evaluate_all, qualitative_examples
from src.preprocessing import truncate
from src.semantic_retriever import (
    DEFAULT_MODEL_NAME,
    SemanticModelUnavailable,
    SemanticRetriever,
    load_sentence_transformer,
)
from src.tfidf_retriever import TFIDFRetriever

# --------------------------------------------------------------------------- #
# Product identity and design tokens
# --------------------------------------------------------------------------- #

APP_NAME = "QuerySense"
APP_TAGLINE = "Intelligent Semantic Search & FAQ Retrieval"

NAVY = "#0f172a"
SLATE = "#475569"
MUTED = "#64748b"
INDIGO = "#4f46e5"
INDIGO_SOFT = "#eef2ff"
BORDER = "#e2e8f0"
SURFACE = "#f8fafc"

MODEL_LABELS = {
    "tfidf": "TF-IDF (lexical baseline)",
    "semantic": "Semantic Search (MiniLM embeddings)",
}

# Heuristic, model-specific "this result is probably too weak to trust" levels.
# These are display heuristics only — they are never used to tune or filter the
# evaluation, and they are not comparable between the two models.
LOW_SIGNAL_THRESHOLD = {"tfidf": 0.12, "semantic": 0.28}

EXAMPLE_QUERIES = [
    "How can I find out whether I am eligible to take my exams?",
    "I forgot my login password for the student website",
    "can I keep a library book longer than two weeks",
    "money back if I leave the course before it starts",
    "how do I get a transcript for a university abroad",
    "electric kettle in my hostel room is it allowed",
]

CUSTOM_CSS = f"""
<style>
  .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }}
  html, body, [class*="css"] {{ color: {NAVY}; }}

  .qs-brand {{ display: flex; align-items: baseline; gap: .6rem; flex-wrap: wrap; }}
  .qs-brand-name {{ font-size: 1.85rem; font-weight: 700; letter-spacing: -.02em; color: {NAVY}; }}
  .qs-brand-tag {{ font-size: .95rem; color: {MUTED}; }}
  .qs-rule {{ height: 1px; background: {BORDER}; margin: .9rem 0 1.6rem 0; }}

  .qs-section {{ font-size: 1.05rem; font-weight: 650; color: {NAVY}; margin: 1.4rem 0 .5rem 0; }}
  .qs-note {{ font-size: .84rem; color: {MUTED}; line-height: 1.55; }}

  .qs-card {{
    border: 1px solid {BORDER}; border-radius: 10px; padding: .85rem 1rem .7rem 1rem;
    background: #ffffff; margin-bottom: .35rem;
  }}
  .qs-card-head {{ display: flex; align-items: center; gap: .55rem; margin-bottom: .35rem; }}
  .qs-rank {{
    font-size: .72rem; font-weight: 700; color: {INDIGO}; background: {INDIGO_SOFT};
    border-radius: 5px; padding: .12rem .45rem;
  }}
  .qs-cat {{
    font-size: .72rem; color: {SLATE}; border: 1px solid {BORDER};
    border-radius: 5px; padding: .1rem .45rem; background: {SURFACE};
  }}
  .qs-score {{ margin-left: auto; font-size: .78rem; color: {MUTED}; font-variant-numeric: tabular-nums; }}
  .qs-question {{ font-size: 1rem; font-weight: 600; color: {NAVY}; line-height: 1.4; }}

  .qs-badge {{
    display: inline-block; font-size: .72rem; font-weight: 600; border-radius: 5px;
    padding: .12rem .45rem; margin-right: .3rem;
  }}
  .qs-badge-shared {{ background: {INDIGO_SOFT}; color: {INDIGO}; }}
  .qs-badge-only {{ background: #fff7ed; color: #9a3412; }}
  .qs-badge-fail {{ background: #fef2f2; color: #991b1b; }}

  .qs-notice {{
    border-left: 3px solid {INDIGO}; background: {INDIGO_SOFT};
    padding: .6rem .85rem; border-radius: 0 6px 6px 0; font-size: .87rem; color: {NAVY};
  }}
  .qs-warn {{
    border-left: 3px solid #d97706; background: #fffbeb;
    padding: .6rem .85rem; border-radius: 0 6px 6px 0; font-size: .87rem; color: #78350f;
  }}
  .qs-empty {{
    border: 1px dashed {BORDER}; border-radius: 10px; padding: 1.1rem;
    text-align: center; color: {MUTED}; font-size: .9rem; background: {SURFACE};
  }}
  section[data-testid="stSidebar"] {{ border-right: 1px solid {BORDER}; }}
</style>
"""


# --------------------------------------------------------------------------- #
# Cached resources — the model and every index are built at most once per key
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner=False)
def load_bundled_faqs() -> pd.DataFrame:
    return load_faqs()


@st.cache_data(show_spinner=False)
def load_bundled_queries() -> pd.DataFrame:
    return load_evaluation_queries()


@st.cache_resource(show_spinner=False)
def get_semantic_model() -> tuple[object | None, str]:
    """Load MiniLM once per session, reporting failure rather than raising."""
    try:
        return load_sentence_transformer(DEFAULT_MODEL_NAME), ""
    except SemanticModelUnavailable as exc:
        return None, str(exc)


@st.cache_resource(show_spinner=False)
def build_tfidf_index(ids: tuple[str, ...], texts: tuple[str, ...]) -> TFIDFRetriever:
    return TFIDFRetriever(list(ids), list(texts)).fit()


@st.cache_resource(show_spinner=False)
def build_semantic_index(ids: tuple[str, ...], texts: tuple[str, ...]) -> SemanticRetriever:
    model, error = get_semantic_model()
    if model is None:
        raise SemanticModelUnavailable(error)
    return SemanticRetriever(list(ids), list(texts), model=model).fit()


@st.cache_data(show_spinner=False)
def run_cached_evaluation(fingerprint: str, k: int, include_semantic: bool) -> dict:
    """Evaluate both models on the bundled labeled query set.

    The fingerprint keys the cache to the knowledge base, so re-running only
    happens when the underlying records actually change.
    """
    faqs = load_bundled_faqs()
    queries = load_bundled_queries()

    ids = tuple(faqs["id"])
    texts = tuple(faqs["search_text"])

    retrievers = {"tfidf": build_tfidf_index(ids, texts)}
    unavailable: dict[str, str] = {}

    if include_semantic:
        try:
            retrievers["semantic"] = build_semantic_index(ids, texts)
        except SemanticModelUnavailable as exc:
            unavailable["semantic"] = str(exc)

    results = evaluate_all(retrievers, queries, k=k)
    return {
        "results": results,
        "unavailable": unavailable,
        "comparison": comparison_table(results, k=k),
        "qualitative": qualitative_examples(results, faqs, k=k),
    }


# --------------------------------------------------------------------------- #
# Session helpers
# --------------------------------------------------------------------------- #


def current_faqs() -> pd.DataFrame:
    """The active knowledge base: an uploaded frame if present, else the bundle."""
    uploaded = st.session_state.get("uploaded_faqs")
    return uploaded if uploaded is not None else load_bundled_faqs()


def current_doc_ids() -> tuple[str, ...]:
    return tuple(current_faqs()["id"])


def current_texts() -> tuple[str, ...]:
    return tuple(current_faqs()["search_text"])


def semantic_ready() -> tuple[bool, str]:
    model, error = get_semantic_model()
    return (model is not None), error


def retrieve(kind: str, query: str, k: int, min_score: float | None = None) -> list[RetrievalHit]:
    """Run one retriever and return ranked hits."""
    ids, texts = current_doc_ids(), current_texts()
    if kind == "tfidf":
        retriever = build_tfidf_index(ids, texts)
    else:
        retriever = build_semantic_index(ids, texts)
    return retriever.search(query, k=k, min_score=min_score)


def push_history(query: str, model_label: str, hits: list[RetrievalHit], k: int) -> None:
    history = st.session_state.setdefault("search_history", [])
    history.insert(
        0,
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "query": query,
            "model": model_label,
            "k": k,
            "results": len(hits),
            "top_score": round(hits[0].score, 3) if hits else None,
            "top_id": hits[0].doc_id if hits else None,
        },
    )
    del history[12:]


def faq_lookup(faqs: pd.DataFrame) -> dict[str, dict]:
    return {row["id"]: row for row in faqs.to_dict("records")}


# --------------------------------------------------------------------------- #
# Presentation helpers
# --------------------------------------------------------------------------- #


def header() -> None:
    st.markdown(
        f"""
        <div class="qs-brand">
          <span class="qs-brand-name">{APP_NAME}</span>
          <span class="qs-brand-tag">{APP_TAGLINE}</span>
        </div>
        <div class="qs-rule"></div>
        """,
        unsafe_allow_html=True,
    )


def note(text: str) -> None:
    st.markdown(f'<div class="qs-note">{text}</div>', unsafe_allow_html=True)


def notice(text: str) -> None:
    st.markdown(f'<div class="qs-notice">{text}</div>', unsafe_allow_html=True)


def warn_box(text: str) -> None:
    st.markdown(f'<div class="qs-warn">{text}</div>', unsafe_allow_html=True)


def empty_box(text: str) -> None:
    st.markdown(f'<div class="qs-empty">{text}</div>', unsafe_allow_html=True)


def section(text: str) -> None:
    st.markdown(f'<div class="qs-section">{text}</div>', unsafe_allow_html=True)


def render_hit_card(hit: RetrievalHit, record: dict, score_label: str, rank: int | None = None) -> None:
    """One ranked answer card with an expandable full answer."""
    st.markdown(
        f"""
        <div class="qs-card">
          <div class="qs-card-head">
            <span class="qs-rank">#{rank if rank is not None else hit.rank}</span>
            <span class="qs-cat">{html.escape(str(record.get("category", "Uncategorized")))}</span>
            <span class="qs-score">{hit.score:.3f} &middot; {html.escape(score_label)}</span>
          </div>
          <div class="qs-question">{html.escape(str(record.get("question", "")))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Retrieved answer & metadata", expanded=False):
        st.write(record.get("answer", ""))
        meta_cols = st.columns(3)
        meta_cols[0].caption(f"FAQ id: {record.get('id', '')}")
        meta_cols[1].caption(f"Source: {record.get('source', '')}")
        meta_cols[2].caption(f"Similarity: {hit.score:.4f}")
        keywords = record.get("keywords", "")
        if keywords:
            st.caption(f"Keywords: {keywords}")
        note(
            "This is the stored knowledge-base record retrieved by similarity search. "
            "It is not a newly generated answer, and the similarity score is not a "
            "probability."
        )


def low_signal_message(best_score: float | None, kind: str) -> None:
    warn_box(
        f"The best match scores only <b>{best_score:.3f}</b>, which is weak for this model. "
        "No stored FAQ answers this question confidently — try a more specific query, "
        "or rephrase using terms from the topic you are looking for."
        if best_score is not None
        else "No stored FAQ matches this query."
    )


def metric_row(labels_values: list[tuple[str, str, str | None]]) -> None:
    columns = st.columns(len(labels_values))
    for column, (label, value, help_text) in zip(columns, labels_values):
        column.metric(label, value, help=help_text)


# --------------------------------------------------------------------------- #
# Page 1 — Search
# --------------------------------------------------------------------------- #


def page_search() -> None:
    header()
    st.markdown('<div class="qs-section">Ask a question in natural language</div>', unsafe_allow_html=True)
    note(
        "QuerySense retrieves the closest stored FAQ records from the knowledge base. "
        "It does not generate new text — every answer shown is retrieved verbatim."
    )

    available, error = semantic_ready()

    with st.form("search_form", clear_on_submit=False):
        query = st.text_input(
            "Your question",
            value=st.session_state.get("search_query", ""),
            placeholder="e.g. How can I find out whether I am eligible to take my exams?",
            label_visibility="collapsed",
        )
        controls = st.columns([2, 1, 1])
        model_choice = controls[0].radio(
            "Retrieval model",
            options=list(MODEL_LABELS),
            format_func=lambda key: MODEL_LABELS[key],
            horizontal=True,
            disabled=not available,
        )
        top_k = controls[1].slider("Results (top-k)", MIN_TOP_K, MAX_TOP_K, DEFAULT_TOP_K)
        use_threshold = controls[2].checkbox("Apply score threshold", value=False)
        threshold = st.slider(
            "Minimum similarity score",
            0.0,
            1.0,
            0.0,
            0.01,
            disabled=not use_threshold,
            help="Hides results scoring below this value for the selected model.",
        )
        submitted = st.form_submit_button("Search", type="primary", use_container_width=False)

    if not available:
        warn_box(
            "<b>Semantic search is unavailable in this environment.</b><br>"
            f"{html.escape(error)}<br>TF-IDF retrieval remains fully functional."
        )

    section("Example questions")
    example = st.pills(
        "Example questions",
        options=EXAMPLE_QUERIES,
        label_visibility="collapsed",
    )

    active_query = None
    if submitted and query.strip():
        active_query = query.strip()
    elif example:
        active_query = example

    if active_query:
        st.session_state["search_query"] = active_query
        applied_threshold = threshold if use_threshold else None
        with st.spinner(f"Retrieving with {MODEL_LABELS[model_choice]}..."):
            hits = retrieve(model_choice, active_query, top_k, applied_threshold)
        st.session_state["last_search"] = {
            "query": active_query,
            "model": model_choice,
            "k": top_k,
            "threshold": applied_threshold,
            "hits": hits,
            "fingerprint": knowledge_base_fingerprint(current_faqs()),
        }
        push_history(active_query, MODEL_LABELS[model_choice], hits, top_k)

    outcome = st.session_state.get("last_search")
    if not outcome:
        empty_box("Enter a question above, or pick an example, to retrieve matching FAQ records.")
    else:
        render_search_outcome(outcome)

    render_history()


def render_search_outcome(outcome: dict) -> None:
    faqs = current_faqs()
    records = faq_lookup(faqs)
    hits = outcome["hits"]
    kind = outcome["model"]

    if outcome["fingerprint"] != knowledge_base_fingerprint(faqs):
        st.info("The knowledge base changed since this search. Run the query again to refresh results.")

    section(f"Results for “{outcome['query']}”")
    st.caption(
        f"Model: {MODEL_LABELS[kind]} · top-{outcome['k']} · "
        f"{len(hits)} result(s)"
        + (f" · min score {outcome['threshold']:.2f}" if outcome["threshold"] else "")
    )

    if not hits:
        low_signal_message(None, kind)
        return

    best = hits[0].score
    if best < LOW_SIGNAL_THRESHOLD[kind]:
        low_signal_message(best, kind)

    for hit in hits:
        record = records.get(hit.doc_id)
        if record is None:
            continue
        render_hit_card(hit, record, MODEL_LABELS[kind].split(" (")[0])

    note(
        "Scores are model-specific cosine similarities over different representations and are "
        "not calibrated probabilities or confidence percentages. A TF-IDF score of 0.5 and an "
        "embedding score of 0.5 do not mean the same thing."
    )


def render_history() -> None:
    history = st.session_state.get("search_history", [])
    section("Session search history")
    if not history:
        note("Searches you run in this session will be listed here.")
        return
    st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
    if st.button("Clear history"):
        st.session_state["search_history"] = []
        st.rerun()


# --------------------------------------------------------------------------- #
# Page 2 — Model Comparison
# --------------------------------------------------------------------------- #


def page_comparison() -> None:
    header()
    st.markdown('<div class="qs-section">Run one query through both retrieval models</div>', unsafe_allow_html=True)
    note(
        "TF-IDF matches surface vocabulary; semantic search matches meaning through sentence "
        "embeddings. Running the same query through both shows where the two disagree."
    )

    available, error = semantic_ready()
    if not available:
        warn_box(
            "<b>Semantic search is unavailable, so a comparison cannot be produced.</b><br>"
            f"{html.escape(error)}"
        )

    with st.form("compare_form"):
        query = st.text_input(
            "Your question",
            placeholder="e.g. money back if I leave the course before it starts",
            label_visibility="collapsed",
        )
        top_k = st.slider("Results per model (top-k)", MIN_TOP_K, MAX_TOP_K, DEFAULT_TOP_K)
        submitted = st.form_submit_button("Compare both models", type="primary", disabled=not available)

    if submitted and query.strip():
        st.session_state["compare_query"] = query.strip()
        st.session_state["compare_k"] = top_k

    active = st.session_state.get("compare_query")
    if not active:
        empty_box("Enter a question to see both models side by side.")
        return
    if not available:
        return

    k = st.session_state.get("compare_k", DEFAULT_TOP_K)
    with st.spinner("Retrieving with both models..."):
        tfidf_hits = retrieve("tfidf", active, k)
        semantic_hits = retrieve("semantic", active, k)

    section(f"Query: “{active}”")
    faqs = current_faqs()
    records = faq_lookup(faqs)

    left, right = st.columns(2)
    with left:
        st.markdown("**TF-IDF (lexical baseline)**")
        if not tfidf_hits:
            low_signal_message(None, "tfidf")
        for hit in tfidf_hits:
            render_hit_card(hit, records[hit.doc_id], "TF-IDF cosine")
    with right:
        st.markdown("**Semantic search (MiniLM embeddings)**")
        if not semantic_hits:
            low_signal_message(None, "semantic")
        for hit in semantic_hits:
            render_hit_card(hit, records[hit.doc_id], "embedding cosine")

    section("Rank changes")
    table = rank_change_table(faqs, tfidf_hits, semantic_hits)
    st.dataframe(table, use_container_width=True, hide_index=True)
    note(
        "Rank change is TF-IDF rank minus semantic rank: a positive value means semantic "
        "retrieval placed the record higher. An em dash means the model did not return that "
        "record at all within the requested top-k."
    )

    section("Shared and model-specific results")
    shared = sorted(set(h.doc_id for h in tfidf_hits) & set(h.doc_id for h in semantic_hits))
    only_tfidf = sorted(set(h.doc_id for h in tfidf_hits) - set(h.doc_id for h in semantic_hits))
    only_semantic = sorted(set(h.doc_id for h in semantic_hits) - set(h.doc_id for h in tfidf_hits))

    st.markdown(
        f'<span class="qs-badge qs-badge-shared">Shared ({len(shared)})</span> '
        + " ".join(f"`{doc_id}`" for doc_id in shared)
        if shared
        else '<span class="qs-badge qs-badge-shared">No shared results</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<span class="qs-badge qs-badge-only">TF-IDF only ({len(only_tfidf)})</span> '
        + " ".join(f"`{doc_id}`" for doc_id in only_tfidf),
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<span class="qs-badge qs-badge-only">Semantic only ({len(only_semantic)})</span> '
        + " ".join(f"`{doc_id}`" for doc_id in only_semantic),
        unsafe_allow_html=True,
    )

    section("Why the two models differ")
    st.markdown(
        """
        - **TF-IDF** scores documents by shared words and word-pairs, weighted so that rare
          terms matter more. If your query uses none of the words in the FAQ, the score is
          zero, even when the meaning is identical — this is why paraphrases are missed.
        - **Sentence embeddings** place a whole sentence into a dense vector space where
          paraphrases sit close together. A query can be retrieved without sharing a single
          keyword with the stored question.
        - **Rank position** is the observable difference. Raw scores are *not* comparable
          across models: each model uses its own geometry, so a 0.62 TF-IDF score and a 0.62
          embedding score mean different things. Compare ranks and retrieved sets, not scores.
        """
    )


def rank_change_table(
    faqs: pd.DataFrame, tfidf_hits: list[RetrievalHit], semantic_hits: list[RetrievalHit]
) -> pd.DataFrame:
    tfidf_rank = {hit.doc_id: hit.rank for hit in tfidf_hits}
    semantic_rank = {hit.doc_id: hit.rank for hit in semantic_hits}
    questions = dict(zip(faqs["id"], faqs["question"]))
    categories = dict(zip(faqs["id"], faqs["category"]))

    ordered = sorted(
        set(tfidf_rank) | set(semantic_rank),
        key=lambda doc_id: (tfidf_rank.get(doc_id, 99), semantic_rank.get(doc_id, 99)),
    )

    rows = []
    for doc_id in ordered:
        in_tfidf = doc_id in tfidf_rank
        in_semantic = doc_id in semantic_rank
        rows.append(
            {
                "FAQ id": doc_id,
                "Question": truncate(questions.get(doc_id, ""), 68),
                "Category": categories.get(doc_id, ""),
                "TF-IDF rank": tfidf_rank.get(doc_id, "—"),
                "Semantic rank": semantic_rank.get(doc_id, "—"),
                "Rank change": (tfidf_rank[doc_id] - semantic_rank[doc_id])
                if (in_tfidf and in_semantic)
                else "—",
                "Retrieved by": "Both models"
                if (in_tfidf and in_semantic)
                else ("TF-IDF only" if in_tfidf else "Semantic only"),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Page 3 — Evaluation Dashboard
# --------------------------------------------------------------------------- #


def page_evaluation() -> None:
    header()
    st.markdown('<div class="qs-section">Offline evaluation on the labeled query set</div>', unsafe_allow_html=True)
    note(
        "Both models are evaluated on the identical knowledge base and the identical labeled "
        "queries. Every number below is computed from real predictions — none of it is hardcoded."
    )

    faqs = load_bundled_faqs()
    queries = load_bundled_queries()
    available, error = semantic_ready()

    controls = st.columns([1, 1, 2])
    k = controls[0].selectbox("Cut-off k", list(range(MIN_TOP_K, MAX_TOP_K + 1)), index=2)
    include_semantic = controls[1].checkbox("Include semantic model", value=available, disabled=not available)
    run = controls[2].button("Run evaluation", type="primary")

    if not available:
        warn_box(
            "<b>Semantic model unavailable — only the TF-IDF baseline can be evaluated.</b><br>"
            f"{html.escape(error)}"
        )
        notice("Semantic results are marked unavailable rather than estimated or fabricated.")

    cache_key = "evaluation_outcome"
    if run:
        with st.spinner("Evaluating both models on the labeled query set..."):
            st.session_state[cache_key] = run_cached_evaluation(
                knowledge_base_fingerprint(faqs), k, include_semantic
            )

    outcome = st.session_state.get(cache_key)
    if not outcome or outcome["results"].get("tfidf", {}).get("summary", {}).get("k") != k:
        empty_box(
            f"No evaluation has been run for k = {k} in this session. "
            "Select a cut-off and press <b>Run evaluation</b>."
        )
        render_evaluation_methodology(k)
        return

    results = outcome["results"]
    unavailable = outcome["unavailable"]

    if "semantic" in unavailable and "semantic" not in results:
        warn_box(f"Semantic evaluation skipped: {html.escape(unavailable['semantic'])}")

    section("Headline metrics")
    primary = results.get("semantic") or results.get("tfidf")
    summary = primary["summary"]

    metric_row(
        [
            (f"Precision@{k}", f"{summary['precision_at_k']:.4f}", "Relevant retrieved ÷ k, macro-averaged"),
            (f"Recall@{k}", f"{summary['recall_at_k']:.4f}", "Relevant retrieved ÷ all relevant, macro-averaged"),
            ("MRR", f"{summary['mrr']:.4f}", "Mean reciprocal rank of the first relevant result"),
            ("Median latency", f"{summary['median_latency_ms']:.2f} ms", "Steady-state per-query latency"),
            ("Queries", f"{summary['evaluated_queries']}", "Number of labeled evaluation queries"),
        ]
    )
    if "semantic" in results:
        st.caption("Headline metrics shown for Semantic Search (MiniLM embeddings).")

    section("Model comparison")
    comparison = outcome["comparison"]
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.plotly_chart(metrics_bar_chart(comparison, k), use_container_width=True)

    section(f"Per-query results (top-{k})")
    per_query_frames = []
    for name, result in results.items():
        frame = result["per_query"][
            [
                "model",
                "query_id",
                "query",
                "query_type",
                "relevant_ids",
                "top_k_ids",
                "precision_at_k",
                "recall_at_k",
                "reciprocal_rank",
                "latency_ms",
            ]
        ].copy()
        per_query_frames.append(frame)
    combined = pd.concat(per_query_frames, ignore_index=True)
    st.dataframe(combined, use_container_width=True, hide_index=True)
    st.download_button(
        "Download per-query results (CSV)",
        combined.to_csv(index=False).encode("utf-8"),
        file_name=f"querysense_evaluation_k{k}.csv",
        mime="text/csv",
    )

    correct_and_missed(results, faqs, k)

    if "semantic" in results and "tfidf" in results:
        section("Qualitative comparison")
        qualitative = outcome["qualitative"]
        if qualitative.empty:
            note("No qualitative comparison available.")
        else:
            st.dataframe(
                qualitative[
                    [
                        "case_label",
                        "query",
                        "relevant_ids",
                        "tfidf_top_ids",
                        "semantic_top_ids",
                        "tfidf_first_relevant_rank",
                        "semantic_first_relevant_rank",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
            note(
                "Cases are grouped by outcome: agreement, ranking divergence, and failure. "
                "A rank of 0 means no relevant record was retrieved in the top-k."
            )

    render_evaluation_methodology(k)


def correct_and_missed(results: dict, faqs: pd.DataFrame, k: int) -> None:
    questions = dict(zip(faqs["id"], faqs["question"]))
    section(f"Correctly retrieved vs missed (top-{k})")

    for name, result in results.items():
        frame = result["per_query"]
        hits = frame[frame["correct"]]
        misses = frame[~frame["correct"]]

        left, right = st.columns(2)
        with left:
            st.markdown(f"**{name}: correctly retrieved — {len(hits)} of {len(frame)} queries**")
            if hits.empty:
                note("No query retrieved a relevant record in the top-k.")
            else:
                preview = pd.DataFrame(
                    {
                        "query": hits["query"].map(lambda value: truncate(value, 58)),
                        "relevant id": hits["relevant_ids"],
                        "first relevant rank": hits["first_relevant_rank"],
                    }
                )
                st.dataframe(preview, use_container_width=True, hide_index=True)
        with right:
            st.markdown(f"**{name}: missed — {len(misses)} of {len(frame)} queries**")
            if misses.empty:
                note("No missed queries.")
            else:
                preview = pd.DataFrame(
                    {
                        "query": misses["query"].map(lambda value: truncate(value, 58)),
                        "missed relevant id": misses["missed_relevant_ids"].map(
                            lambda value: value or "—"
                        ),
                    }
                )
                st.dataframe(preview, use_container_width=True, hide_index=True)

        top_miss = misses.head(3)
        for row in top_miss.itertuples(index=False):
            relevant = row.relevant_ids.split("|")
            explanations = "; ".join(
                f"{doc_id}: {truncate(questions.get(doc_id, 'unknown record'), 80)}"
                for doc_id in relevant
            )
            st.caption(f"Missed for “{truncate(row.query, 70)}” — relevant record(s) → {explanations}")


def metrics_bar_chart(comparison: pd.DataFrame, k: int) -> go.Figure:
    metrics = [f"Precision@{k}", f"Recall@{k}", "MRR"]
    figure = go.Figure()
    colors = {"tfidf": "#94a3b8", "semantic": INDIGO}
    for row in comparison.to_dict("records"):
        model = row["Model"]
        figure.add_trace(
            go.Bar(
                name=MODEL_LABELS.get(model, model),
                x=metrics,
                y=[row[metric] for metric in metrics],
                marker_color=colors.get(model, "#64748b"),
                text=[f"{row[metric]:.3f}" for metric in metrics],
                textposition="outside",
            )
        )
    figure.update_layout(
        barmode="group",
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(title=f"Macro-averaged score (k = {k})", range=[0, 1], gridcolor=BORDER),
        xaxis=dict(title=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return figure


def render_evaluation_methodology(k: int) -> None:
    with st.expander("How these numbers are produced"):
        st.markdown(
            f"""
            - **Ground truth**: `data/evaluation_queries.csv` holds {len(load_bundled_queries())}
              natural-language queries with manually assigned relevant FAQ ids. Labels were written
              by hand and are independent of any model prediction.
            - **Precision@{k}** = relevant records in the top-{k} ÷ {k}. The denominator is always
              {k}, so a query with no relevant result scores 0 rather than being undefined.
            - **Recall@{k}** = relevant records in the top-{k} ÷ the total number of relevant
              records labelled for that query.
            - **MRR** = mean of 1 ÷ the rank of the first relevant result, or 0 when no relevant
              record is retrieved.
            - **Latency** is measured with `time.perf_counter()` per query in a steady-state pass.
              Model loading and index construction are timed separately and excluded; a warm-up
              pass runs first so the reported latency is not first-call overhead.
            - Both models see the identical knowledge base and identical queries. No threshold or
              model was tuned on these results.
            """
        )


# --------------------------------------------------------------------------- #
# Page 4 — Knowledge Base
# --------------------------------------------------------------------------- #


def page_knowledge_base() -> None:
    header()
    st.markdown('<div class="qs-section">Knowledge base</div>', unsafe_allow_html=True)
    notice(SAMPLE_DATA_NOTICE)

    faqs = current_faqs()
    source_label = "Uploaded in this session" if st.session_state.get("uploaded_faqs") is not None else "Bundled sample dataset"

    metric_row(
        [
            ("Records", f"{len(faqs)}", "FAQ records currently indexed"),
            ("Categories", f"{faqs['category'].nunique()}", "Distinct categories"),
            ("Active source", source_label, "Uploads live in session state only"),
            ("Fingerprint", knowledge_base_fingerprint(faqs), "Index cache key"),
        ]
    )

    section("Category distribution")
    distribution = category_distribution(faqs)
    figure = go.Figure(
        go.Bar(
            x=distribution["records"],
            y=distribution["category"],
            orientation="h",
            marker_color=INDIGO,
            text=distribution["records"],
            textposition="outside",
        )
    )
    figure.update_layout(
        height=max(300, 34 * len(distribution)),
        margin=dict(l=10, r=20, t=10, b=10),
        xaxis=dict(title="Records", gridcolor=BORDER),
        yaxis=dict(title="", autorange="reversed"),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(figure, use_container_width=True)

    section("Browse and inspect")
    filters = st.columns([2, 2])
    category = filters[0].selectbox("Category filter", ["All categories"] + sorted(faqs["category"].unique()))
    search_term = filters[1].text_input("Filter by text in question, answer or keywords", "")

    view = faqs if category == "All categories" else faqs[faqs["category"] == category]
    if search_term.strip():
        needle = search_term.strip().lower()
        haystack = (
            view["question"].str.lower()
            + " "
            + view["answer"].str.lower()
            + " "
            + view["keywords"].str.lower()
        )
        view = view[haystack.str.contains(needle, regex=False)]

    st.caption(f"{len(view)} of {len(faqs)} records shown")
    st.dataframe(
        view[["id", "category", "question", "keywords", "source"]],
        use_container_width=True,
        hide_index=True,
    )

    section("Inspect a record")
    if len(view) == 0:
        note("No records match the current filter.")
    else:
        selected = st.selectbox("FAQ id", view["id"].tolist())
        record = view[view["id"] == selected].iloc[0].to_dict()
        st.markdown(f"**{record['question']}**")
        st.write(record["answer"])
        st.caption(
            f"id: {record['id']} · category: {record['category']} · "
            f"source: {record['source']} · keywords: {record['keywords']}"
        )
        st.caption(f"Indexed search text: {record['search_text']}")

    section("Upload a CSV knowledge base")
    note(
        "Required columns: <code>question</code> and <code>answer</code>. Optional: "
        "<code>id</code> (generated when missing), <code>category</code>, "
        "<code>keywords</code>, <code>source</code>. Uploaded data is held in session state and "
        "<b>never overwrites the bundled sample data on disk</b>."
    )
    uploaded = st.file_uploader("Upload FAQ CSV", type=["csv"])
    actions = st.columns([1, 1, 3])

    if actions[0].button("Use uploaded CSV", disabled=uploaded is None):
        try:
            frame = load_faqs_from_csv(uploaded.getvalue())
        except DataValidationError as exc:
            st.error(f"Upload rejected: {exc}")
        else:
            st.session_state["uploaded_faqs"] = frame
            st.success(
                f"Loaded {len(frame)} records. Indexes and embeddings will be rebuilt for the new "
                "knowledge base."
            )
            st.rerun()

    if actions[1].button("Restore sample dataset"):
        st.session_state.pop("uploaded_faqs", None)
        st.success("Restored the bundled 150-record sample dataset.")
        st.rerun()

    with st.expander("Download the bundled sample dataset"):
        st.download_button(
            "Download faqs.csv",
            load_bundled_faqs()[["id", "category", "question", "answer", "keywords", "source"]]
            .to_csv(index=False)
            .encode("utf-8"),
            file_name="faqs.csv",
            mime="text/csv",
        )


# --------------------------------------------------------------------------- #
# Page 5 — About & Methodology
# --------------------------------------------------------------------------- #


def page_about() -> None:
    header()
    st.markdown('<div class="qs-section">About & methodology</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
**Problem statement.** Students ask questions in their own words, but institutional knowledge
sits in a fixed FAQ list. Keyword search fails as soon as the wording differs; this project
measures how much a semantic retriever closes that gap.

**What QuerySense is.** An information-retrieval system. It ranks and returns stored FAQ
records. It does **not** generate new answers, and no retrieved text is presented as a
newly written response.

**NLP pipeline**

1. **Load and validate** the knowledge base (`question`, `answer` + optional metadata), rejecting
   duplicate or missing ids and empty fields.
2. **Build the searchable representation** from question + keywords + category. Answers stay
   attached as retrieved content but are not indexed, so long answers cannot dominate scoring.
3. **TF-IDF baseline** — `sklearn.TfidfVectorizer` (1–2 grams, sublinear TF, English stop words)
   producing L2-normalized sparse vectors; cosine similarity via the dot product.
4. **Semantic retrieval** — `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional, CPU
   inference) with normalized embeddings; cosine similarity via the dot product.
5. **Rank, filter and display** the top-k with model-specific similarity scores.

**TF-IDF versus sentence embeddings**

| | TF-IDF | Sentence embeddings |
|---|---|---|
| Representation | Sparse counts over n-grams | Dense 384-d context vector |
| Matches on | Shared surface words | Meaning |
| Paraphrase "attendance … write exams" → "allowed to sit for papers" | Usually misses (no shared vocabulary) | Usually retrieves |
| Needs training data | No | No — pretrained, used as-is |
| Query latency | Sub-millisecond | Several milliseconds |

**Dataset provenance.** `data/faqs.csv` contains 150 synthetic, illustrative FAQ records across
7 categories, written for this capstone. `data/evaluation_queries.csv` contains 40
natural-language queries with manual relevance labels, deliberately written so that no query
repeats a FAQ's exact wording. Both files ship with the repository, so no download is required
and results are reproducible. The data does **not** describe the official policies of any real
institution. No public dataset was substituted, because none with independent ground-truth
relevance labels could be adopted within the project's time budget without changing the task.

**Evaluation methodology.** Both models run over the identical knowledge base and the identical
40 labeled queries. Metrics are macro-averaged over queries; latency is measured with
`time.perf_counter()` in a steady-state pass after a warm-up, with model loading and index
construction timed separately. The evaluation set is used once for reporting — no thresholds
were tuned on it, and no model was selected using these results.

**Limitations**

- The corpus is small (150 records) and synthetic, so absolute metric values are not
  representative of a production FAQ system.
- Relevance labels are single-annotator; a second annotator would reveal label ambiguity.
- A cosine similarity score measures *similarity*, not factual correctness — the top-ranked
  record can be topically related yet not answer the question.
- Some queries are genuinely ambiguous or split across several records, so a low score can
  reflect the label design rather than a model weakness.
- Semantic retrieval costs roughly an order of magnitude more latency than TF-IDF.

**Reproducing these results**

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python -m pytest -q                 # 114 tests, no network required
python -m src.evaluation --k 3      # writes results/metrics.json + CSVs
streamlit run app.py                # launch this interface
```

Model: `{DEFAULT_MODEL_NAME}` · Similarity: cosine · Embedding: 384-d, CPU inference
"""
    )


# --------------------------------------------------------------------------- #
# Shell
# --------------------------------------------------------------------------- #


PAGES = {
    "Search": page_search,
    "Model Comparison": page_comparison,
    "Evaluation Dashboard": page_evaluation,
    "Knowledge Base": page_knowledge_base,
    "About & Methodology": page_about,
}


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(f"### {APP_NAME}")
        st.caption(APP_TAGLINE)
        st.divider()

        page = st.radio("Navigation", list(PAGES), label_visibility="collapsed")
        st.divider()

        faqs = current_faqs()
        st.markdown("**Knowledge base**")
        st.caption(
            f"{len(faqs)} records · {faqs['category'].nunique()} categories · "
            f"{'uploaded' if st.session_state.get('uploaded_faqs') is not None else 'bundled sample'}"
        )

        available, _ = semantic_ready()
        st.markdown("**Models**")
        st.caption("TF-IDF baseline · available")
        st.caption(
            "MiniLM semantic · " + ("available" if available else "unavailable")
        )

        st.divider()
        st.caption(
            "Illustrative demo data. Retrieved answers are stored FAQ records, not "
            "generated text, and are not official institutional policy."
        )
    return page


def main() -> None:
    st.set_page_config(page_title=f"{APP_NAME} · FAQ Retrieval", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.session_state.setdefault("search_history", [])

    page = render_sidebar()
    PAGES[page]()


if __name__ == "__main__":
    main()
