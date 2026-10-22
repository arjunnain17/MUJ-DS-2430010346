# QuerySense — Capstone Project Summary

**Project:** QuerySense — Intelligent Semantic Search & FAQ Retrieval System
**Registration number:** `MUJ-DS-2430010346`
**Batch:** Batch F
**Repository:** https://github.com/arjunnain17/MUJ-DS-2430010346
**Student name:** `<YOUR FULL NAME>` · **Branch:** `<YOUR BRANCH>` · **Program:** `<PROGRAM>`

> Placeholders above must be filled in before submission. No personal details were invented.

---

## 1. Executive summary

QuerySense is an end-to-end NLP information-retrieval system that answers natural-language
questions by ranking and returning records from a structured FAQ knowledge base. It implements
two retrieval strategies — a **TF-IDF + cosine similarity baseline** and **semantic retrieval
using `sentence-transformers/all-MiniLM-L6-v2`** — and evaluates both on the same 150-record
knowledge base and the same 40 hand-labeled queries.

**Headline outcome at k = 3:**

| Metric | TF-IDF | Semantic | Difference |
|---|---|---|---|
| Precision@3 | 0.2833 | **0.3500** | +0.0667 |
| Recall@3 | 0.6625 | **0.8250** | +0.1625 |
| MRR | 0.6542 | **0.8583** | **+0.2042** |
| Median query latency | **0.80 ms** | 10.94 ms | +10.14 ms |
| Queries with no relevant result in top 3 | 9 / 40 | **3 / 40** | −6 queries |

Semantic retrieval improves MRR by roughly **31 % relative** and reduces total retrieval
failures by two thirds, at roughly **an order of magnitude more per-query latency**. The project
also documents the cases where semantic retrieval loses, and the cases where both models fail.

> **Reproducibility:** ranking metrics are deterministic and reproduce exactly. Query latency is
> not — across three runs on the same machine TF-IDF median latency ranged 0.66–1.02 ms and
> semantic 9.57–11.11 ms. `results/metrics.json` is the source of truth.

---

## 2. Scope delivered against the brief

| Requirement | Status | Evidence |
|---|---|---|
| Self-contained dataset, no manual download (100–200 records) | ✅ | `data/faqs.csv`, 150 records, 7 categories |
| Labeled evaluation set, ≥ 30 queries, independent of predictions | ✅ | `data/evaluation_queries.csv`, 40 queries |
| TF-IDF + cosine baseline | ✅ | `src/tfidf_retriever.py`, 2,144-term vocabulary |
| Semantic retrieval with `all-MiniLM-L6-v2`, CPU | ✅ | `src/semantic_retriever.py`, 384-d normalized |
| Configurable top-k (1–10, default 3) | ✅ | `src/base_retriever.py::clamp_top_k` |
| Precision@3, Recall@3, MRR, mean/median latency | ✅ | `src/metrics.py`, `src/evaluation.py` |
| Same queries and KB for both models | ✅ | asserted in `tests/test_evaluation.py` |
| Metrics from real predictions, never hardcoded | ✅ | `results/metrics.json`, degrade-and-compare test |
| Metric chart + comparison table | ✅ | `results/`, Evaluation Dashboard page |
| ≥ 5 qualitative examples incl. failures | ✅ | `results/qualitative_examples.csv`, 40 rows across 5 case types |
| Five-page Streamlit interface | ✅ | `app.py` |
| Cached model/embeddings; rebuild only on KB change | ✅ | fingerprint-keyed `st.cache_resource` / `st.cache_data` |
| Graceful model-download failure with TF-IDF fallback | ✅ | `SemanticModelUnavailable`, verified by tests |
| CSV upload with validation, never overwrites the bundle | ✅ | `load_faqs_from_csv`, 7 dedicated tests |
| pytest suite, executed | ✅ | **114 tests, all passing, offline** |
| README, presentation outline, summary | ✅ | `README.md`, `presentations/`, `capstone/` |
| Screenshots of the working app | ✅ | `screenshots/`, 9 real browser captures |
| Git commits for milestones, pushed to origin | ✅ | 6 milestone commits on `main` |
| Not an LLM chatbot | ✅ | retrieval only; no generated text anywhere |

**Not delivered / deferred:** hybrid (fused) retrieval, FAISS indexing and cross-encoder
reranking are documented as future work. A browser-driven end-to-end test of the CSV upload
control was not performed (the upload path is covered by unit tests instead).

---

## 3. Method

**Data.** 150 synthetic, illustrative FAQ records across 7 categories (fees/scholarships 24,
hostel/campus 22, registration/exams 22, technical support 22, admissions 20, attendance policy
20, library 20). Each record carries `id`, `category`, `question`, `answer`, `keywords`,
`source`. The data is generic institutional content: **it is not, and is not presented as, the
official policy of any real institution.**

**Preprocessing.** Whitespace normalization, missing-value handling, unique-id enforcement,
keyword tokenization, and construction of one searchable string per record from
question + keywords + category. **Answers are deliberately excluded from the index** so long
answers cannot dominate similarity scores; they remain attached as retrieved content. No
stemming or stop-word removal is applied to the embedding input, to avoid degrading semantics.

**Retrieval.** Both models reduce to cosine similarity over L2-normalized vectors, which makes
the computation directly comparable while keeping the score *interpretations* distinct:

- TF-IDF: `TfidfVectorizer` (1–2 grams, sublinear TF, English stop words, 2,144 features).
- Semantic: `all-MiniLM-L6-v2`, 384-d normalized embeddings, CPU inference, no fine-tuning.

Ranking is deterministic (score descending, then document index) and non-positive scores are
dropped, so a query with no lexical overlap returns no hits rather than misleading ones.

**Evaluation.** Both models over the identical KB and identical queries. Precision@k uses k as
the denominator; Recall@k divides by the labeled relevant count; MRR uses the reciprocal rank of
the first relevant result (0 if none). Ranking metrics are macro-averaged over queries. Latency
is measured with `time.perf_counter()` in a steady-state pass after a warm-up, with model load
and index build timed separately and excluded. No parameter was tuned on the evaluation set.

---

## 4. Key findings

1. **Paraphrase handling is the decisive difference.** For
   *"money back if I leave the course before it starts"*, TF-IDF returned three *leave-of-absence*
   FAQs (reciprocal rank 0.0) while semantic retrieval returned the refund-policy record at rank 1
   (1.0), with **zero overlap** between the two result sets.
2. **Semantic retrieval is better on balance, not universally.** Of 40 queries: 19 both models
   succeeded at the same rank, 11 both succeeded with different rankings, **7 semantic-only**,
   **1 TF-IDF-only**, and **2 both failed**.
3. **Failure modes are diagnosable, not mysterious.** The both-fail cases
   (*"can I stay somewhere on campus instead of travelling daily"*, *"how many days can I keep a
   library book"*) reflect knowledge-base coverage and label design as much as model weakness —
   the intent is expressed situationally while the FAQ uses domain vocabulary.
4. **The quality gain has a real cost.** ~10× median latency (10.94 ms vs 0.80 ms) plus a one-time
   ~10 s model load. Both are invisible at this scale; neither would be at millions of records.
5. **Precision@3 is structurally capped low.** Many queries have 1–2 relevant records, so a
   perfect retrieval scores at most 0.333. Recall@3 and MRR are the informative metrics on this set.

---

## 5. Engineering quality

- **114 automated tests, all passing, fully offline.** Semantic code paths are exercised through
  an injected deterministic fake encoder, so the suite needs no model download and no network.
- Test coverage spans schema and upload validation, duplicate/missing ids, empty and malformed
  CSVs, empty and zero-overlap queries, score ranges, deterministic ranking, shared document ids
  across both retrievers, model-download failure handling, and hand-computed Precision/Recall/MRR
  including no-relevant-result cases.
- **Anti-fabrication test:** `test_metrics_are_not_hardcoded` degrades the retriever's index and
  asserts the reported MRR actually falls, proving metrics derive from predictions.
- **Graceful degradation:** if the embedding model cannot load, semantic mode is disabled with an
  explicit reason and TF-IDF continues to work; the evaluation records semantic results as
  unavailable rather than estimating them.
- **One real defect found and fixed** during browser verification: the app inherited Streamlit's
  dark theme while its stylesheet assumed a light background, producing unreadable contrast.
  The theme is now pinned in `.streamlit/config.toml`.

---

## 6. How to run

```bash
cd QuerySense
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python -m pytest -q                # 114 tests, no network required
python -m src.evaluation --k 3     # regenerates results/
streamlit run app.py               # http://localhost:8501
```

---

## 7. Limitations

- Small synthetic corpus (150 records): relative model behaviour is meaningful, absolute metric
  values are not production-representative.
- Query latency is noisy between runs, unlike the deterministic ranking metrics.
- Single-annotator relevance labels; inter-annotator agreement not measured.
- Cosine similarity measures similarity, not factual correctness.
- Ambiguous and multi-record queries inflate measured error; part of it is label design.
- Two models' scores come from different geometries and are never compared as calibrated values.
- English-only, no institutional domain adaptation.

---

## 8. Future work

Scale the corpus with multiple annotators; add **hybrid retrieval via reciprocal rank fusion** as
a third evaluated arm; replace brute-force similarity with FAISS/HNSW and re-measure latency;
add cross-encoder reranking of the top 20; replace the fixed similarity threshold with a learned
"no good answer" classifier; and run a human answer-correctness study distinct from ranking metrics.
