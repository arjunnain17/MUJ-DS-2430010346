# QuerySense — Presentation Outline (5–7 minutes)

**Format:** ~12 slides, live demo in the middle.
**Golden rule for the demo:** never claim a number you did not generate in this session.

---

## Slide 1 — Title (15 s)

**QuerySense — Intelligent Semantic Search & FAQ Retrieval**
Subtitle: Comparing TF-IDF and sentence-embedding retrieval on a labeled FAQ query set

Also show: Student name · Registration number `MUJ-DS-2430010346` · Branch · Batch F · date

> **Say:** "This is a retrieval system, not a chatbot. It returns stored FAQ records — it does
> not generate text."

---

## Slide 2 — The problem (45 s)

Put one contrast on the screen, large:

- User asks: **"money back if I leave the course before it starts"**
- FAQ says: **"What is the refund policy if I withdraw my admission before classes start?"**

Shared content words: **almost none.**

Then the punchline: TF-IDF returns three *leave-of-absence* FAQs — confident, and completely
wrong. The word "leave" is doing all the work.

> **Say:** "Keyword search assumes the user knows our vocabulary. Students don't."

**Visual:** the two sentences side by side, with the shared-word mismatch highlighted.

---

## Slide 3 — Objectives (30 s)

Four bullets only:

1. Build a validated FAQ knowledge base that ships with the repo.
2. Implement a TF-IDF + cosine baseline.
3. Implement semantic retrieval with `all-MiniLM-L6-v2` embeddings on CPU.
4. **Measure the gap** on the same queries with Precision@k, Recall@k, MRR and latency.

> **Say:** "The deliverable is the measurement, not just the demo."

---

## Slide 4 — The knowledge base (30 s)

- **150** FAQ records across **7** categories (admissions, registration/exams, attendance
  policy, library, hostel/campus, technical support, fees/scholarships).
- Each record: `id`, `category`, `question`, `answer`, `keywords`, `source`.
- **40** evaluation queries with hand-assigned relevant ids — paraphrases, short queries,
  keyword-heavy queries and multi-concept queries.

**Say this clearly:** "This data is **synthetic and illustrative**. It is not the official
policy of any real institution, and I do not present it as such."

---

## Slide 5 — Pipeline (45 s)

Show the architecture diagram from the README.

Walk the arrow path once, out loud:

1. Validate + normalize the CSV → reject duplicate ids and empty fields.
2. Build the **searchable text** from question + keywords + category.
   → **Key design decision:** answers are *excluded* from the index, so a long answer cannot
     dominate the similarity score.
3. Two parallel retrievers → both reduce to cosine similarity: sparse TF-IDF vectors, and
   L2-normalized 384-d MiniLM embeddings.
4. Rank deterministically, filter by optional threshold, display top-k.

> **Say:** "Both models produce a cosine similarity — which makes the scores mechanically
> comparable to compute, but **not** comparable to interpret."

---

## Slide 6 — Why the two models differ (40 s)

| | TF-IDF | Sentence embeddings |
|---|---|---|
| Representation | Sparse n-gram counts | Dense 384-d vector |
| Matches on | Shared words | Meaning |
| "attendance … write exams" → "allowed to sit for papers" | Usually misses | Usually retrieves |
| Training needed | None (statistical fit) | None (pretrained, used as-is) |
| Query latency | < 1 ms | ~10 ms |

> **Say:** "If the query and the document share no tokens, TF-IDF similarity is exactly zero —
> no amount of cleverness fixes that. Embeddings don't have that constraint."

---

## Slide 7 — LIVE DEMO: the failure case (60 s) ⭐

**Do this exact sequence.**

1. Open **Model Comparison**.
2. Type: `money back if I leave the course before it starts`
3. Press **Compare both models**.

Point at the screen:

- **Left (TF-IDF):** three Attendance & Academic Policy records — "inform the department before
  taking leave", "medical leave certificate", "leave of absence". **None relevant.**
- **Right (Semantic):** rank 1 is *"What is the refund policy if I withdraw my admission before
  classes start?"* — the correct answer.
- **"No shared results"** — the two models returned *disjoint* sets.

> **Say:** "Same query, same knowledge base, zero overlap. That's the whole project in one screen."

**Fallback if the semantic model is unavailable:** run the Search page in TF-IDF mode, then show
`results/qualitative_examples.csv` for the recorded semantic output and state plainly that the
live semantic model could not be loaded in this environment.

---

## Slide 8 — Evaluation methodology (40 s)

Emphasise the rigor, briefly:

- Both models on the **same** 150 records and the **same** 40 labeled queries.
- **Precision@3** = relevant in top-3 ÷ **3** (denominator is always k).
- **Recall@3** = relevant in top-3 ÷ all relevant for that query.
- **MRR** = mean of 1 ÷ rank of first relevant result.
- Latency via `time.perf_counter()`, **excluding** model load and index build, after a warm-up pass.
- **No threshold or model was tuned on the test set.** Labels are independent of predictions.
- **Nothing is hardcoded** — if the model can't load, results are marked unavailable, not invented.

---

## Slide 9 — Results (45 s)

Show the comparison table and the bar chart.

| Model | Precision@3 | Recall@3 | MRR | Median latency |
|---|---|---|---|---|
| TF-IDF (baseline) | 0.2833 | 0.6625 | 0.6542 | 0.80 ms |
| **Semantic (MiniLM)** | **0.3500** | **0.8250** | **0.8583** | 10.94 ms |
| Difference | +0.0667 | +0.1625 | **+0.2042** | +10.14 ms |

Plus: queries where nothing relevant was retrieved in the top 3 → **TF-IDF 9/40 vs semantic 3/40.**

> **Say:** "MRR improves by about 31 % relative. The cost is roughly an order of magnitude more
> latency — about 11 milliseconds instead of under 1, which is invisible in this application but
> would matter at scale."

**If asked about latency stability:** ranking metrics reproduce exactly; latency does not. Across
three runs on the same machine, TF-IDF median latency moved 0.66–1.02 ms and semantic
9.57–11.11 ms. `results/metrics.json` is the source of truth.

**If asked why Precision@3 looks low:** many queries have only one relevant record, so a perfect
retrieval caps out at 1/3 = 0.333. Recall@3 and MRR are the informative metrics here.

---

## Slide 10 — Qualitative analysis (40 s)

Five cases, one line each (full detail in `results/qualitative_examples.csv`):

| Case | Query | Outcome |
|---|---|---|
| Paraphrase trap | "money back if I leave the course…" | TF-IDF fails (0.0), semantic rank 1 |
| Lexical trap | "electric kettle in my hostel room…" | TF-IDF returns hostel *maintenance/fees*; semantic rank 1 |
| Zero overlap | "installment option for paying admission charges" | TF-IDF misses; semantic rank 1 |
| **TF-IDF wins** | "i failed a subject what do i do next" | TF-IDF rank 1; semantic rank 1 wrong |
| **Both fail** | "can I stay somewhere on campus instead of travelling daily" | neither finds the hostel record |

**7 queries: only semantic succeeded. 1 query: only TF-IDF succeeded. 2 queries: both failed.**

> **Say:** "Semantic retrieval is better on balance, not universally. I'm showing the cases where
> it lost too."

---

## Slide 11 — Limitations & future work (40 s)

**Limitations (say honestly):**
- 150 synthetic records — relative behaviour is meaningful, absolute values are not production-grade.
- Single-annotator labels; no inter-annotator agreement measured.
- Similarity ≠ factual correctness; a top result can be topically adjacent yet not answer the question.
- Ambiguous/under-specified queries inflate the error (partly a label-design issue).
- Scores from the two models are **not** calibrated and are never compared as if they were.

**Future work:**
- Scale to 1,000+ records with two annotators.
- **Hybrid retrieval (reciprocal rank fusion)** as a third evaluated arm.
- FAISS/HNSW index + cross-encoder reranking of the top 20.
- Learn a "no good answer" classifier instead of a fixed similarity threshold.

---

## Slide 12 — Engineering & close (30 s)

- **114 automated tests, all passing, fully offline** (fake encoder injected for semantic tests;
  no network needed). Covers schema validation, upload rejection, hand-computed
  Precision/Recall/MRR, deterministic ranking, and model-download failure handling.
- Reproducible in three commands:
  ```bash
  pip install -r requirements.txt
  python -m pytest -q
  python -m src.evaluation --k 3
  ```
- One real defect found during browser verification and fixed: the app inherited Streamlit's dark
  theme while its stylesheet assumed light, producing unreadable contrast — now pinned in
  `.streamlit/config.toml`.

> **Closing line:** "QuerySense shows that on this FAQ set, semantic retrieval closes a real and
> measurable gap in paraphrase handling — at roughly ten times the query latency — and it shows
> exactly where that improvement still runs out."

---

## Anticipated questions

**"Why not just use an LLM to answer?"**
That changes the task. This is information retrieval: the answer must be traceable to a stored
record. An LLM would introduce hallucination risk and would no longer be evaluable with
Precision@k / MRR against relevance labels.

**"Is Precision@3 = 0.35 bad?"**
It's capped by construction. Most queries have 1–2 relevant records, so the maximum achievable
Precision@3 is 0.333–0.667. MRR (0.858) and Recall@3 (0.825) are the meaningful numbers.

**"Why is semantic latency an order of magnitude higher?"**
It encodes the query through a 6-layer transformer (~11 ms on CPU). TF-IDF is a sparse
matrix-vector product under 1 ms. Both are negligible here; the gap matters only at scale.

**"Could you combine them?"**
Yes — that's the top item in future work. Reciprocal rank fusion typically beats either model
alone. I did not add it because the brief asked for a baseline-versus-semantic comparison.

**"Why are the similarity scores different scales?"**
Different geometries entirely. TF-IDF scores are sparse-cosine over shared n-grams; embedding
scores are dense-cosine in a learned 384-d space. I report them as model-specific similarities
and compare *ranks*, never raw scores.

**"How do you know the metrics aren't fake?"**
Every number comes from `python -m src.evaluation`, written to `results/metrics.json` with the
knowledge-base fingerprint and environment recorded. `test_metrics_are_not_hardcoded` degrades
the retriever and asserts the reported MRR actually drops.
