# MUJ-DS-2430010346 — Data Science Capstone

**QuerySense — Intelligent Semantic Search & FAQ Retrieval System**

An end-to-end NLP information-retrieval project that answers natural-language questions by
ranking and returning records from a structured FAQ knowledge base, and empirically compares two
retrieval strategies:

1. **Baseline** — TF-IDF vectorization with cosine similarity (lexical matching).
2. **Semantic retrieval** — Sentence Transformers (`sentence-transformers/all-MiniLM-L6-v2`)
   with cosine similarity over dense embeddings.

On the bundled 150-record knowledge base and 40 hand-labeled evaluation queries at k = 3:

| Model | Precision@3 | Recall@3 | MRR | Median latency |
|---|---|---|---|---|
| TF-IDF (baseline) | 0.2833 | 0.6625 | 0.6542 | 0.80 ms |
| **Semantic (MiniLM)** | **0.3500** | **0.8250** | **0.8583** | 10.94 ms |

Semantic retrieval improves MRR by roughly **31 % relative** and cuts retrieval failures from
9/40 queries to 3/40, at roughly an order of magnitude more per-query latency. Ranking metrics
are exactly reproducible; latency varies between runs (see the full README).

> This is a retrieval system, not a generative chatbot. Every answer shown is a stored FAQ
> record returned verbatim. The bundled data is **synthetic and illustrative** and is not the
> official policy of any real institution.

## Quick start

```bash
cd QuerySense
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python -m pytest -q                # 114 tests, fully offline
python -m src.evaluation --k 3     # regenerates results/
streamlit run app.py               # http://localhost:8501
```

## Project layout

```
MUJ-DS-2430010346/
└── QuerySense/          # the capstone project (see its README for full documentation)
    ├── app.py           # Streamlit interface (5 pages)
    ├── data/            # 150 illustrative FAQs + 40 labeled evaluation queries
    ├── src/             # data loading, preprocessing, both retrievers, metrics, evaluation
    ├── tests/           # 114 tests (offline)
    ├── notebooks/       # reproducible experiments walkthrough
    ├── results/         # metrics.json, evaluation_results.csv, qualitative_examples.csv
    ├── screenshots/     # captures of the running application
    ├── presentations/   # 5-7 minute demo outline
    └── capstone/        # project summary
```

## Documentation

- **[Full project README](QuerySense/README.md)** — problem statement, architecture, dataset
  provenance, models, evaluation methodology, actual results, limitations, troubleshooting.
- **[Project summary](QuerySense/capstone/project_summary.md)** — scope delivered against the
  brief, key findings and engineering quality.
- **[Presentation outline](QuerySense/presentations/presentation_outline.md)** — 5–7 minute
  demonstration script with anticipated questions.
- **[Experiments notebook](QuerySense/notebooks/experiments.ipynb)** — end-to-end walkthrough.

## Licence

MIT — see [QuerySense/LICENSE](QuerySense/LICENSE).

## Student details

| Field | Value |
|---|---|
| Student name | `<YOUR FULL NAME>` |
| Registration number | `MUJ-DS-2430010346` |
| Branch | `<YOUR BRANCH>` |
| Batch | `Batch F` |
| GitHub | `arjunnain17` |
| Training program | `<PROGRAM NAME>` |

> Placeholders must be filled in before submission. No personal information was invented.
