# ArXiv RAG Platform

A self-hosted Retrieval-Augmented Generation system for ArXiv research papers. Search and chat with 200+ papers from cs.LG, cs.AI, and cs.CL, with a RAGAS evaluation suite that compares naive and reranked retrieval using a free local judge model.

**Runs entirely via `docker compose up`. No cloud infrastructure required.**

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        User (browser)                           │
│                     arxiv.localhost (React)                     │
└─────────────────┬───────────────────────────────────────────────┘
                  │ HTTP
┌─────────────────▼───────────────────────────────────────────────┐
│              Backend (FastAPI)  :8000                           │
│   /search  /chat  /ingest  /eval/run  /eval/history             │
│                                                                 │
│   Embed query ──► Qdrant ANN ──► Cross-encoder rerank ──► LLM  │
│   (MiniLM-L6-v2)   (top-20)    (ms-marco-MiniLM)    (GPT/Ollama)│
└──────┬────────────────┬──────────────────────┬──────────────────┘
       │                │                      │
┌──────▼──────┐  ┌──────▼──────┐  ┌────────────▼──────────────────┐
│   Qdrant    │  │  Postgres   │  │   Evaluation Service  :8002   │
│   :6333     │  │  :5432      │  │   RAGAS runner, MLflow logger  │
│  (vectors)  │  │ (query logs │  └────────────┬──────────────────┘
└─────────────┘  │  eval runs) │               │
                 └─────────────┘  ┌────────────▼──────┐
                                  │  MLflow :5000     │
┌─────────────────────────────────│  (tracking server)│
│  Ingestion Service  :8001       └───────────────────┘
│  ArXiv API ► PyMuPDF ► chunk ► embed ► upsert
│  APScheduler re-runs daily
└─────────────────────────────────────────────────────
```

### Stack

| Component | Technology |
|-----------|-----------|
| Vector DB | Qdrant v1.9 |
| Embeddings | `all-MiniLM-L6-v2` (local, no API) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) |
| LLM | GPT-4o-mini (OpenAI) or Ollama llama3 (free fallback) |
| Evaluation | RAGAS 0.4 (faithfulness, answer relevancy, context precision, context recall), judged by a local Ollama model |
| Experiment tracking | MLflow |
| Database | PostgreSQL 16 |
| Backend | FastAPI + Python 3.11 |
| Frontend | React 18 + Vite + Tailwind CSS |
| PDF parsing | PyMuPDF |
| Scheduling | APScheduler |

---

## Quickstart

### Prerequisites

- Docker Desktop (or Docker Engine + Compose plugin)
- 8 GB RAM recommended (models download on first build)
- [Ollama](https://ollama.ai) running locally with `qwen2.5:7b` pulled (evaluation judge) and `llama3` pulled (free generation fallback)
- _Optional_: OpenAI API key for GPT-4o-mini generation

### 1. Clone and configure

```bash
git clone <repo-url>
cd arxiv-rag
cp .env.example .env
# Edit .env and set OPENAI_API_KEY= (or leave blank to use Ollama)
```

### 2. Start everything

```bash
docker compose up --build
```

First run takes 10–20 minutes because:
- Docker builds all images and downloads base layers
- Sentence-transformer and cross-encoder models are downloaded during build (~500 MB)
- Ingestion service fetches and indexes 210 ArXiv papers (3 categories × 70 papers)

**The backend won't start until ingestion is healthy** — `depends_on` with healthchecks ensures correct startup order.

### 3. Access the services

| Service | URL |
|---------|-----|
| Frontend (Search / Chat / Eval) | http://arxiv.localhost |
| Backend API docs | http://arxiv.localhost/api/docs |
| MLflow tracking UI | http://localhost:5001 |
| Qdrant dashboard | http://localhost:6333/dashboard |

---

## How to Run Evaluations

### Via the frontend

First pull the judge model: `ollama pull qwen2.5:7b`.

Navigate to http://arxiv.localhost/eval, select a pipeline variant, optionally add notes, and click **Run Eval**. A run takes several minutes with a local judge.

### Via curl

```bash
# Run naive (no reranker) evaluation
curl -X POST http://arxiv.localhost/api/eval/run \
  -H "Content-Type: application/json" \
  -d '{"pipeline_variant": "naive", "notes": "baseline comparison"}'

# Run full pipeline (with reranker)
curl -X POST http://arxiv.localhost/api/eval/run \
  -H "Content-Type: application/json" \
  -d '{"pipeline_variant": "reranked", "notes": "full pipeline"}'

# Fetch history
curl http://arxiv.localhost/api/eval/history
```

### What's being evaluated

The test set (`services/evaluation/testset.json`) contains 25 question/answer pairs spanning transformers, alignment, inference optimization, RAG, and more. Each run writes a JSON result file to `evals/results/` with per-question scores, the judge and generator models, and any failures. See [EVALUATION.md](EVALUATION.md) for the metrics, the judge setup, and how to read the results.

> Scoring uses a local judge (`qwen2.5:7b` via Ollama by default), so it is free. Any OpenAI-compatible endpoint can be the judge instead; set `JUDGE_BASE_URL`, `JUDGE_MODEL`, and `JUDGE_API_KEY`.

---

## How to Add New Paper Categories

1. Edit `.env` (or `docker-compose.yml` environment):

```bash
ARXIV_CATEGORIES=cs.LG,cs.AI,cs.CL,cs.CV,stat.ML
PAPERS_PER_CATEGORY=70
```

2. Trigger re-ingestion:

```bash
curl -X POST http://arxiv.localhost/api/ingest
```

Or restart the ingestion container:

```bash
docker compose restart ingestion
```

3. The ingestion service deduplicates by `arxiv_id` — already-indexed papers are skipped, only new ones are added.

Available ArXiv categories: `cs.CV` (computer vision), `cs.RO` (robotics), `cs.NE` (neural/evolutionary computation), `stat.ML` (statistics / ML), `eess.SP` (signal processing), etc.

---

## Demo Walkthrough

### Search

> Query: *"attention mechanism self-attention transformers"*

The semantic search returns the top-10 most relevant paper chunks, deduplicated to unique papers with relevance scores. Each card shows title, authors, abstract excerpt, and links to ArXiv and PDF.

### Chat (with source citations)

> Question: *"How does LoRA reduce the cost of fine-tuning large language models?"*

**Example answer** (illustrative, not a recorded output):
> LoRA reduces fine-tuning cost by injecting trainable low-rank decomposition matrices (A and B where W = W₀ + BA) into the attention weight matrices, while keeping the original pretrained weights frozen. This reduces trainable parameters by 10,000× compared to full fine-tuning. [Source 1] shows LoRA achieves comparable performance to full fine-tuning on GPT-3 with only 0.01% of the parameters updated. [Source 2] demonstrates that rank r=8 is sufficient for most downstream tasks.

**Sources** (collapsible):
- [1] *LoRA: Low-Rank Adaptation of Large Language Models* — Hu et al. (2022)
- [2] *LLaMA: Open and Efficient Foundation Language Models* — Touvron et al. (2023)

### Eval Dashboard

The dashboard shows a table of all past RAGAS evaluation runs with scores for each metric, color-coded by quality (green ≥ 75%, yellow ≥ 55%, red < 55%). A line chart shows score trends across runs.

---

## Project Structure

```
arxiv-rag/
├── docker-compose.yml          # Full stack definition with healthchecks
├── .env.example                # Environment template
├── README.md
├── EVALUATION.md               # Benchmark results and analysis
├── postgres/
│   └── init.sql                # Schema: query_logs, eval_results
├── qdrant/
│   └── config.yaml
├── services/
│   ├── ingestion/              # ArXiv fetch + chunk + embed + upsert
│   │   ├── main.py             # FastAPI app + APScheduler
│   │   ├── ingestion.py        # Core pipeline
│   │   ├── settings.py
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   ├── backend/                # Query orchestration
│   │   ├── main.py             # FastAPI: /search /chat /eval/*
│   │   ├── rag.py              # Retrieve → rerank → generate
│   │   ├── database.py         # Postgres logging
│   │   ├── models.py           # Pydantic schemas
│   │   ├── settings.py
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   └── evaluation/             # RAGAS runner (local judge)
│       ├── main.py             # FastAPI app
│       ├── evaluator.py        # Run, score, record (JSON, MLflow, Postgres)
│       ├── judge.py            # RAGAS metrics + judge endpoint config
│       ├── tests/              # pytest suite
│       ├── settings.py
│       ├── testset.json        # 25 ground-truth Q&A pairs
│       ├── requirements.txt
│       └── Dockerfile
└── frontend/                   # React + Vite + Tailwind
    ├── src/
    │   ├── App.jsx
    │   ├── pages/
    │   │   ├── SearchPage.jsx
    │   │   ├── ChatPage.jsx
    │   │   └── EvalPage.jsx
    │   ├── components/
    │   │   ├── PaperCard.jsx
    │   │   └── SourceCitation.jsx
    │   └── hooks/useApi.js
    ├── Dockerfile
    ├── nginx.conf
    └── package.json
```

---

## Useful Commands

```bash
# Tail all logs
docker compose logs -f

# Check ingestion progress
docker compose logs -f ingestion

# Rebuild a single service after code changes
docker compose up --build backend

# Stop everything (preserves volumes)
docker compose down

# Full reset (deletes all indexed papers and eval history)
docker compose down -v

# Manual ingest trigger
curl -X POST http://arxiv.localhost/api/ingest

# Check how many papers are indexed
docker compose exec ingestion curl -s localhost:8001/status
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | _(empty)_ | If set, uses GPT-4o-mini for chat generation |
| `OLLAMA_URL` | `http://host.docker.internal:11434` | Ollama endpoint (used when no OpenAI key) |
| `OLLAMA_MODEL` | `llama3` | Ollama model to use |
| `ARXIV_CATEGORIES` | `cs.LG,cs.AI,cs.CL` | Comma-separated ArXiv category list |
| `PAPERS_PER_CATEGORY` | `70` | Papers to fetch per category per run |
| `RUN_EVAL_ON_STARTUP` | `false` | Auto-run evaluation suite on first start |
| `JUDGE_MODEL` | `qwen2.5:7b` | Evaluation judge model |
| `JUDGE_BASE_URL` | `{OLLAMA_URL}/v1` | OpenAI-compatible judge endpoint |
| `JUDGE_API_KEY` | _(empty)_ | Only for hosted judge endpoints |
| `JUDGE_CONCURRENCY` | `2` | Questions scored in parallel |

---

## Free vs Paid Operation

| Feature | Free (Ollama) | Paid (OpenAI) |
|---------|:---:|:---:|
| Semantic search | ✓ | ✓ |
| Chat with papers | ✓* | ✓ |
| RAGAS evaluation | ✓ (local judge) | ✓ (local judge) |
| Embeddings | ✓ local | ✓ local |
| Reranking | ✓ local | ✓ local |

*Requires Ollama running on host with `ollama pull llama3`
