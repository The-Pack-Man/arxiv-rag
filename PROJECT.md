---
name: ArXiv RAG Platform
repo: The-Pack-Man/arxiv-rag
status: active
one_liner: Self-hosted RAG search and chat over ArXiv ML papers, with a naive-vs-reranked evaluation harness.
problem: Comparing retrieval strategies needs a repeatable evaluation loop, not one-off demos.
user: Personal research use; portfolio demonstration.
tags: [eval-reliability, observability-data, shipped-evidence]
stack: [Python, FastAPI, Qdrant, sentence-transformers, cross-encoder reranking, RAGAS, PostgreSQL, MLflow, React, Vite, Docker Compose, Traefik, Ollama, OpenAI API, pytest]
run: docker compose up --build
test: cd services/evaluation && pytest

claims:
  - id: C1
    text: Built a multi-service RAG stack (ingestion, query backend, evaluation, frontend) orchestrated with Docker Compose and health-checked startup ordering.
    evidence: [docker-compose.yml, services/]
  - id: C2
    text: Implemented two-stage retrieval, with dense ANN over Qdrant (top-20) followed by cross-encoder reranking (top-5), switchable per request.
    evidence: [services/backend/rag.py]
  - id: C3
    text: Built a scheduled ingestion pipeline (ArXiv API, PyMuPDF parsing, token-window chunking with overlap, embedding, upsert) with arxiv_id deduplication.
    evidence: [services/ingestion/ingestion.py, services/ingestion/main.py]
  - id: C4
    text: Built a RAGAS 0.4 evaluation service (faithfulness, answer relevancy, context precision, context recall) that scores naive vs reranked retrieval on a 25-question ground-truth set, using a free local judge model via any OpenAI-compatible endpoint (Ollama by default).
    evidence: [services/evaluation/evaluator.py, services/evaluation/judge.py, services/evaluation/testset.json, EVALUATION.md]
  - id: C7
    text: Made judge failures visible. A failed or NaN metric is recorded as null and counted per metric rather than replaced with a default, and every run writes a reproducible JSON artifact (judge, generator, per-question scores) plus MLflow and Postgres records.
    evidence: [services/evaluation/evaluator.py, evals/results/]
  - id: C8
    text: Fixed evaluation to score the full chunks the LLM actually received instead of 300-character UI excerpts.
    evidence: [services/backend/models.py, services/backend/main.py]
  - id: C9
    text: Tested the evaluator with pytest, including an integration test that runs the real RAGAS metrics against a local fake OpenAI-compatible judge server.
    evidence: [services/evaluation/tests/]
  - id: C10
    text: Measured naive vs reranked retrieval with paired, per-question RAGAS comparisons and bootstrap confidence intervals, and wrote up the result including the failure mode where reranking amplifies corpus gaps.
    evidence: [scripts/compare_runs.py, EVALUATION.md, evals/results/]
  - id: C5
    text: Logged every query with latency and model used to Postgres for later analysis.
    evidence: [services/backend/database.py, postgres/init.sql]
  - id: C6
    text: Added an LLM fallback path (OpenAI to local Ollama) and degraded-mode responses when generation fails.
    evidence: [services/backend/rag.py]

metrics:
  - id: M1
    name: Evaluator test suite passing
    value: 6
    status: measured
    method: pytest in services/evaluation (Python 3.11), 2026-10-17
    artifact: ""
    date: 2026-10-17
  - id: M2
    name: Context precision, naive vs cross-encoder reranked retrieval (RAGAS, 25 questions)
    value: "0.30 -> 0.50 (+0.20, 95% paired bootstrap interval +0.09 to +0.31)"
    status: measured
    method: RAGAS 0.4, judge qwen2.5:7b, generator llama3, local Ollama, 551 papers indexed, one run; scripts/compare_runs.py
    artifact: evals/results/20261018T221520Z_reranked.json
    date: 2026-10-18
  - id: M3
    name: Faithfulness, answer relevancy, context recall, naive vs reranked
    value: "0.77 vs 0.81, 0.82 vs 0.77, 0.57 vs 0.58; all within noise (intervals include zero)"
    status: measured
    method: same runs as M2
    artifact: evals/results/20261018T215227Z_naive.json
    date: 2026-10-18

decisions:
  - Replaced fabricated benchmark numbers with measured ones and reported where the claim reversed (answer relevancy +19% claimed, -0.05 measured, within noise).
  - Compared variants question by question with bootstrap intervals rather than reporting means alone.
  - Local embedding and reranking models keep the stack free to run. Only generation optionally uses a paid API.
  - The reranker is a request flag, so both variants run through the identical backend path during evaluation.

limitations:
  - 25 questions, one run, 7B judge; only the context-precision gain is distinguishable from noise.
  - Context recall (~0.58) is limited by corpus coverage: ingestion pulls recent papers, and many test questions concern classic papers that may not be indexed.
  - A 7B local judge is noisier than a frontier model. Compare variants under the same judge only.
  - Backend, ingestion, and frontend have no automated tests. Only the evaluator does.
  - Single-node, local deployment. No auth, rate limiting, or load testing.
---

## Context
A testbed for comparing retrieval strategies on a technical corpus, with evaluation built in from the start.

## Results
None yet. Run the evaluation locally (see EVALUATION.md), commit the JSON files under evals/results/, then add the scores to `metrics` with their artifact paths.
