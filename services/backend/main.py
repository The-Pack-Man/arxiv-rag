"""
ArXiv RAG Backend
Endpoints: /health /search /chat /ingest /eval/run /eval/history
"""
import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any

import httpx
import mlflow
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from qdrant_client import QdrantClient

from database import get_eval_history, log_query, save_eval_result
from models import (
    ChatRequest,
    ChatResponse,
    EvalResult,
    EvalRunRequest,
    SearchRequest,
    SearchResponse,
    SourceCitation,
    PaperResult,
)
from rag import embed_query, retrieve, rerank, run_rag_pipeline
from settings import Settings

settings = Settings()
qdrant_client: QdrantClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global qdrant_client
    logger.info("Connecting to Qdrant...")
    qdrant_client = QdrantClient(url=settings.qdrant_url, timeout=60)
    logger.info("Connected to Qdrant.")
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("arxiv-rag-eval")
    yield
    if qdrant_client:
        qdrant_client.close()


app = FastAPI(title="ArXiv RAG API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "model_backend": "openai" if settings.openai_api_key else "ollama"}


@app.post("/ingest")
async def trigger_ingest():
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.post("http://ingestion:8001/ingest")
            return resp.json()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Ingestion service unreachable: {e}")


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Qdrant not ready")

    candidates = retrieve(req.query, qdrant_client, settings, top_k=req.top_k)

    # Deduplicate by arxiv_id, keep highest score per paper
    seen: dict[str, dict[str, Any]] = {}
    for c in candidates:
        aid = c["payload"].get("arxiv_id", "")
        if aid not in seen or c["score"] > seen[aid]["score"]:
            seen[aid] = c

    results = []
    for c in list(seen.values())[: req.top_k]:
        p = c["payload"]
        results.append(
            PaperResult(
                arxiv_id=p.get("arxiv_id", ""),
                title=p.get("title", ""),
                authors=p.get("authors", []),
                abstract=p.get("abstract", ""),
                url=p.get("url", ""),
                pdf_url=p.get("pdf_url", ""),
                published=p.get("published", ""),
                score=c["score"],
                chunk_text=p.get("chunk_text", ""),
            )
        )

    return SearchResponse(results=results, query=req.query, total=len(results))


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Qdrant not ready")

    result = await run_rag_pipeline(
        query=req.query,
        qdrant_client=qdrant_client,
        settings=settings,
        use_reranker=req.use_reranker,
    )

    background_tasks.add_task(
        log_query,
        settings.postgres_url,
        req.query,
        result["answer"],
        result["sources"],
        result["latency_ms"],
        result["model"],
        settings.top_k_rerank,
        result["reranked"],
    )

    return ChatResponse(
        answer=result["answer"],
        sources=[SourceCitation(**s) for s in result["sources"]],
        model=result["model"],
        latency_ms=result["latency_ms"],
        reranked=result["reranked"],
    )


@app.post("/eval/run")
async def run_eval(req: EvalRunRequest, background_tasks: BackgroundTasks):
    """Proxy eval request to evaluation service."""
    async with httpx.AsyncClient(timeout=600) as client:
        try:
            resp = await client.post(
                "http://evaluation:8002/eval/run",
                json=req.model_dump(),
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Eval service error: {e}")


@app.get("/eval/history", response_model=list[EvalResult])
def eval_history(limit: int = 50):
    rows = get_eval_history(settings.postgres_url, limit=limit)
    results = []
    for r in rows:
        params = r.get("params") or {}
        if isinstance(params, str):
            params = json.loads(params)
        results.append(
            EvalResult(
                run_id=r.get("mlflow_run_id") or "",
                pipeline_variant=r.get("pipeline_variant") or "",
                faithfulness=r.get("faithfulness"),
                answer_relevancy=r.get("answer_relevancy"),
                context_precision=r.get("context_precision"),
                context_recall=r.get("context_recall"),
                num_questions=r.get("num_questions") or 0,
                params=params,
                created_at=str(r.get("created_at") or ""),
                notes=r.get("notes") or "",
            )
        )
    return results
