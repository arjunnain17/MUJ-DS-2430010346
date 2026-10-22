"""Reproducible evaluation pipeline comparing retrieval strategies.

Both models are evaluated over the *same* knowledge base and the *same* labeled
query set. Every number reported here comes from real predictions compared
against the manual relevance labels in ``data/evaluation_queries.csv`` — nothing
is hardcoded, and no threshold or model is tuned on these results.

Timing methodology: model loading and index construction are recorded
separately from steady-state query latency. Queries are issued once as a warm-up
pass (untimed) before the timed pass, so the reported latency reflects repeated
use rather than first-call overhead.
"""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from . import __version__
from .base_retriever import DEFAULT_TOP_K, BaseRetriever, clamp_top_k
from .data_loader import (
    DEFAULT_EVAL_PATH,
    DEFAULT_FAQ_PATH,
    knowledge_base_fingerprint,
    load_evaluation_queries,
    load_faqs,
)
from .metrics import (
    average_precision,
    first_relevant_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    summarize_run,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _run_queries_timed(
    retriever: BaseRetriever,
    queries: Sequence[str],
    k: int,
    *,
    warmup: bool = True,
) -> list[tuple[list[str], float]]:
    """Search every query once (optionally warming up) and time only the real pass."""
    if warmup:
        for query in queries:
            retriever.search(query, k=k)

    outcomes: list[tuple[list[str], float]] = []
    for query in queries:
        start = time.perf_counter()
        hits = retriever.search(query, k=k)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        outcomes.append(([hit.doc_id for hit in hits], elapsed_ms))
    return outcomes


def evaluate_retriever(
    retriever: BaseRetriever,
    queries: pd.DataFrame,
    *,
    k: int = DEFAULT_TOP_K,
    warmup: bool = True,
    model_load_seconds: float | None = None,
) -> dict:
    """Evaluate one retriever over the labeled query set.

    Returns ``{"summary": {...}, "per_query": DataFrame}``.
    """
    if queries is None or queries.empty:
        raise ValueError("Cannot evaluate on an empty query set.")

    k = clamp_top_k(k)

    if not retriever.is_fitted:
        retriever.fit()

    query_texts = queries["query"].astype(str).tolist()
    outcomes = _run_queries_timed(retriever, query_texts, k, warmup=warmup)

    rows: list[dict] = []
    latencies: list[float] = []

    for row, (retrieved_ids, latency_ms) in zip(queries.itertuples(index=False), outcomes):
        relevant_ids = list(getattr(row, "relevant_id_list", []))
        top_k_ids = retrieved_ids[:k]

        rank = first_relevant_rank(top_k_ids, relevant_ids)
        rows.append(
            {
                "model": retriever.name,
                "query_id": getattr(row, "query_id", ""),
                "query": getattr(row, "query", ""),
                "query_type": getattr(row, "query_type", "unspecified"),
                "relevant_ids": "|".join(relevant_ids),
                "retrieved_ids": "|".join(retrieved_ids),
                "retrieved_ids_list": retrieved_ids,
                "top_k_ids": "|".join(top_k_ids),
                "precision_at_k": precision_at_k(top_k_ids, relevant_ids, k),
                "recall_at_k": recall_at_k(top_k_ids, relevant_ids, k),
                "reciprocal_rank": reciprocal_rank(top_k_ids, relevant_ids),
                "average_precision": average_precision(top_k_ids, relevant_ids, k),
                "first_relevant_rank": rank if rank is not None else 0,
                "hit_at_k": 1 if rank is not None else 0,
                "correct": bool(rank is not None),
                "missed_relevant_ids": "|".join(
                    doc_id for doc_id in relevant_ids if doc_id not in top_k_ids
                ),
                "latency_ms": latency_ms,
            }
        )
        latencies.append(latency_ms)

    per_query = pd.DataFrame(rows)
    summary = summarize_run(rows, k=k, latencies_ms=latencies)

    load_seconds = (
        float(getattr(retriever, "model_load_seconds", 0.0))
        if model_load_seconds is None
        else float(model_load_seconds)
    )
    summary.update(
        {
            "model": retriever.name,
            "model_description": retriever.describe(),
            "model_load_seconds": round(load_seconds, 4),
            "index_build_seconds": round(float(retriever.build_seconds), 4),
        }
    )
    return {"summary": summary, "per_query": per_query}


def evaluate_all(
    retrievers: Mapping[str, BaseRetriever],
    queries: pd.DataFrame,
    *,
    k: int = DEFAULT_TOP_K,
    warmup: bool = True,
) -> dict[str, dict]:
    """Evaluate several retrievers on the identical query set and knowledge base."""
    if not retrievers:
        raise ValueError("No retrievers were supplied for evaluation.")
    return {
        name: evaluate_retriever(retriever, queries, k=k, warmup=warmup)
        for name, retriever in retrievers.items()
    }


def comparison_table(results: Mapping[str, dict], *, k: int = DEFAULT_TOP_K) -> pd.DataFrame:
    """Side-by-side headline metrics for each model."""
    rows = []
    for name, result in results.items():
        summary = result["summary"]
        rows.append(
            {
                "Model": name,
                f"Precision@{k}": round(summary["precision_at_k"], 4),
                f"Recall@{k}": round(summary["recall_at_k"], 4),
                "MRR": round(summary["mrr"], 4),
                f"MAP@{k}": round(summary["map_at_k"], 4),
                "Mean latency (ms)": round(summary["mean_latency_ms"], 3),
                "Median latency (ms)": round(summary["median_latency_ms"], 3),
                "Queries evaluated": summary["evaluated_queries"],
            }
        )
    return pd.DataFrame(rows)


def _classify_case(tfidf_row: dict, semantic_row: dict) -> str:
    """Label how the two models behave on one query."""
    tfidf_hit = bool(tfidf_row["hit_at_k"])
    semantic_hit = bool(semantic_row["hit_at_k"])

    if tfidf_hit and semantic_hit:
        if tfidf_row["first_relevant_rank"] == semantic_row["first_relevant_rank"]:
            return "both_succeed_same_rank"
        return "both_succeed_rank_differs"
    if semantic_hit and not tfidf_hit:
        return "semantic_only"
    if tfidf_hit and not semantic_hit:
        return "tfidf_only"
    return "both_fail"


CASE_TYPE_LABELS = {
    "both_succeed_same_rank": "Both models succeed (same first relevant rank)",
    "both_succeed_rank_differs": "Both succeed, ranking differs",
    "semantic_only": "Only semantic retrieval succeeds",
    "tfidf_only": "Only TF-IDF succeeds",
    "both_fail": "Both models fail to retrieve a relevant record",
}


def qualitative_examples(
    results: Mapping[str, dict],
    faqs: pd.DataFrame,
    *,
    k: int = DEFAULT_TOP_K,
    per_case_limit: int = 2,
) -> pd.DataFrame:
    """Build qualitative comparison rows covering agreement, divergence and failure."""
    tfidf = results.get("tfidf")
    semantic = results.get("semantic")
    if tfidf is None or semantic is None:
        return pd.DataFrame()

    question_by_id = dict(zip(faqs["id"], faqs["question"]))

    semantic_by_id = {row["query_id"]: row for row in semantic["per_query"].to_dict("records")}

    rows: list[dict] = []
    for tfidf_row in tfidf["per_query"].to_dict("records"):
        semantic_row = semantic_by_id.get(tfidf_row["query_id"])
        if semantic_row is None:
            continue

        case = _classify_case(tfidf_row, semantic_row)
        rows.append(
            {
                "case_type": case,
                "case_label": CASE_TYPE_LABELS[case],
                "query_id": tfidf_row["query_id"],
                "query": tfidf_row["query"],
                "query_type": tfidf_row["query_type"],
                "relevant_ids": tfidf_row["relevant_ids"],
                "tfidf_top_ids": tfidf_row["top_k_ids"],
                "tfidf_first_relevant_rank": tfidf_row["first_relevant_rank"],
                "tfidf_reciprocal_rank": round(tfidf_row["reciprocal_rank"], 4),
                "semantic_top_ids": semantic_row["top_k_ids"],
                "semantic_first_relevant_rank": semantic_row["first_relevant_rank"],
                "semantic_reciprocal_rank": round(semantic_row["reciprocal_rank"], 4),
                "tfidf_top_question": question_by_id.get(
                    (tfidf_row["retrieved_ids_list"] or [""])[0], ""
                ),
                "semantic_top_question": question_by_id.get(
                    (semantic_row["retrieved_ids_list"] or [""])[0], ""
                ),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    table["_order"] = table["case_type"].map(
        {name: index for index, name in enumerate(CASE_TYPE_LABELS)}
    )
    table = table.sort_values(["_order", "query_id"]).drop(columns=["_order"])

    # Keep a balanced, presentation-ready selection: up to N queries per case type.
    curated = (
        table.groupby("case_type", group_keys=False, sort=False)
        .head(per_case_limit)
        .reset_index(drop=True)
    )
    curated["selected_for_report"] = True
    table["selected_for_report"] = table.index.isin(curated["query_id"].index)
    return table.reset_index(drop=True)


def _aggregate_notes(results: Mapping[str, dict], *, k: int) -> list[str]:
    notes = [
        "Precision@k uses k as the denominator, so a query with no relevant result scores 0.",
        "Scores produced by the two retrievers are model-specific and are not calibrated "
        "probabilities; they must not be compared across models.",
        "Both models were evaluated on the identical knowledge base and identical queries, "
        "with no threshold tuning on the evaluation set.",
    ]
    if "tfidf" in results and "semantic" in results:
        tfidf = results["tfidf"]["summary"]
        semantic = results["semantic"]["summary"]
        delta_mrr = semantic["mrr"] - tfidf["mrr"]
        direction = "higher" if delta_mrr > 0 else ("equal" if delta_mrr == 0 else "lower")
        notes.append(
            f"Semantic MRR is {abs(delta_mrr):.4f} {direction} than TF-IDF MRR on this query set."
        )
    if "semantic" not in results:
        notes.append(
            "Semantic results are unavailable in this run; only the TF-IDF baseline was "
            "evaluated. No semantic numbers were estimated or fabricated."
        )
    return notes


def build_metrics_payload(
    results: Mapping[str, dict],
    faqs: pd.DataFrame,
    queries: pd.DataFrame,
    *,
    k: int,
    unavailable: Mapping[str, str] | None = None,
) -> dict:
    """Assemble the machine-readable results object written to results/metrics.json."""
    models: dict[str, dict] = {}
    for name, result in results.items():
        summary = dict(result["summary"])
        summary.pop("model_description", None)
        models[name] = {
            "available": True,
            "description": result["summary"].get("model_description", {}),
            "metrics": summary,
        }

    for name, reason in (unavailable or {}).items():
        if name not in models:
            models[name] = {"available": False, "reason": reason}

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "querysense_version": __version__,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "top_k": k,
        "knowledge_base": {
            "records": int(len(faqs)),
            "categories": int(faqs["category"].nunique()),
            "fingerprint": knowledge_base_fingerprint(faqs),
        },
        "evaluation_set": {
            "queries": int(len(queries)),
            "query_types": queries["query_type"].value_counts().to_dict()
            if "query_type" in queries.columns
            else {},
        },
        "models": models,
        "notes": _aggregate_notes(results, k=k),
    }

    if "tfidf" in results and "semantic" in results:
        tfidf = results["tfidf"]["summary"]
        semantic = results["semantic"]["summary"]
        payload["comparison"] = {
            "precision_at_k_delta": round(semantic["precision_at_k"] - tfidf["precision_at_k"], 4),
            "recall_at_k_delta": round(semantic["recall_at_k"] - tfidf["recall_at_k"], 4),
            "mrr_delta": round(semantic["mrr"] - tfidf["mrr"], 4),
            "median_latency_ms_delta": round(
                semantic["median_latency_ms"] - tfidf["median_latency_ms"], 3
            ),
        }
    return payload


def save_results(
    results: Mapping[str, dict],
    faqs: pd.DataFrame,
    queries: pd.DataFrame,
    *,
    k: int = DEFAULT_TOP_K,
    output_dir: str | Path = RESULTS_DIR,
    qualitative: pd.DataFrame | None = None,
    unavailable: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    """Write metrics.json, evaluation_results.csv and qualitative_examples.csv."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    payload = build_metrics_payload(results, faqs, queries, k=k, unavailable=unavailable)
    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    per_query_frames = [result["per_query"] for result in results.values()]
    if per_query_frames:
        combined = pd.concat(per_query_frames, ignore_index=True)
        export_columns = [
            "model",
            "query_id",
            "query",
            "query_type",
            "relevant_ids",
            "retrieved_ids",
            "precision_at_k",
            "recall_at_k",
            "reciprocal_rank",
            "first_relevant_rank",
            "hit_at_k",
            "correct",
            "missed_relevant_ids",
            "latency_ms",
        ]
        results_path = output / "evaluation_results.csv"
        combined[export_columns].to_csv(results_path, index=False)
    else:
        results_path = output / "evaluation_results.csv"
        results_path.write_text("", encoding="utf-8")

    if qualitative is None:
        qualitative = qualitative_examples(results, faqs, k=k)
    qualitative_path = output / "qualitative_examples.csv"
    qualitative.to_csv(qualitative_path, index=False)

    return {"metrics": metrics_path, "results": results_path, "qualitative": qualitative_path}


def run_full_evaluation(
    *,
    k: int = DEFAULT_TOP_K,
    faq_path: str | Path = DEFAULT_FAQ_PATH,
    query_path: str | Path = DEFAULT_EVAL_PATH,
    output_dir: str | Path = RESULTS_DIR,
    include_semantic: bool = True,
) -> dict:
    """End-to-end evaluation used by ``python -m src.evaluation``.

    Missing semantic support is reported explicitly rather than worked around.
    """
    from .tfidf_retriever import TFIDFRetriever

    faqs = load_faqs(faq_path)
    queries = load_evaluation_queries(query_path)

    doc_ids = faqs["id"].tolist()
    texts = faqs["search_text"].tolist()

    retrievers: dict[str, BaseRetriever] = {
        "tfidf": TFIDFRetriever(doc_ids, texts).fit(),
    }
    unavailable: dict[str, str] = {}

    if include_semantic:
        from .semantic_retriever import SemanticModelUnavailable, SemanticRetriever

        try:
            # Let the retriever load the model itself so that model-load time is
            # recorded separately from query latency and embedding construction.
            retrievers["semantic"] = SemanticRetriever(doc_ids, texts).fit()
        except SemanticModelUnavailable as exc:
            unavailable["semantic"] = str(exc)

    results = evaluate_all(retrievers, queries, k=k)
    qualitative = qualitative_examples(results, faqs, k=k)
    paths = save_results(
        results,
        faqs,
        queries,
        k=k,
        output_dir=output_dir,
        qualitative=qualitative,
        unavailable=unavailable,
    )

    return {
        "results": results,
        "qualitative": qualitative,
        "paths": paths,
        "unavailable": unavailable,
        "comparison": comparison_table(results, k=k),
        "faqs": faqs,
        "queries": queries,
    }


def _main() -> None:  # pragma: no cover - CLI convenience
    import argparse

    parser = argparse.ArgumentParser(description="Run the QuerySense evaluation pipeline.")
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K, help="top-k cutoff (1-10)")
    parser.add_argument("--no-semantic", action="store_true", help="evaluate the TF-IDF baseline only")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    outcome = run_full_evaluation(
        k=args.k,
        output_dir=args.output_dir,
        include_semantic=not args.no_semantic,
    )

    print("\nQuerySense evaluation")
    print("=" * 62)
    print(outcome["comparison"].to_string(index=False))

    for name, reason in outcome["unavailable"].items():
        print(f"\n[{name.upper()} UNAVAILABLE] {reason}")

    print("\nArtifacts written:")
    for label, path in outcome["paths"].items():
        print(f"  {label:12s} {path}")


if __name__ == "__main__":  # pragma: no cover
    _main()
