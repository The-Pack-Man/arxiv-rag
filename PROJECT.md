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
  # Add naive-vs-reranked RAGAS scores here only after a run's JSON is committed under evals/results/.

decisions:
  - Local embedding and reranking models keep the stack free to run. Only generation optionally uses a paid API.
  - The reranker is a request flag, so both variants run through the identical backend path during evaluation.

limitations:
  - No benchmark results committed yet. Scores need a local run with Ollama.
  - A 7B local judge is noisier than a frontier model. Compare variants under the same judge only.
  - Backend, ingestion, and frontend have no automated tests. Only the evaluator does.
  - Single-node, local deployment. No auth, rate limiting, or load testing.
---

## Context
A testbed for comparing retrieval strategies on a technical corpus, with evaluation built in from the start.

## Results
None yet. Run the evaluation locally (see EVALUATION.md), commit the JSON files under evals/results/, then add the scores to `metrics` with their artifact paths.
