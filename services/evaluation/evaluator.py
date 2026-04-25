"""
RAGAS evaluation runner.
Compares naive RAG (no reranker) vs full pipeline (with reranker).
Logs results to MLflow and Postgres.
"""
import json
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import mlflow
import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

from settings import Settings


def load_testset(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


async def query_backend(
    backend_url: str,
    question: str,
    use_reranker: bool,
    timeout: int = 120,
) -> dict[str, Any]:
    """Call backend /chat endpoint."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{backend_url}/chat",
            json={"query": question, "use_reranker": use_reranker},
        )
        resp.raise_for_status()
        return resp.json()


def _ask_ollama(prompt: str, ollama_url: str, model: str) -> str:
    """Send a single prompt to Ollama and return the response text."""
    import httpx
    try:
        resp = httpx.post(
            f"{ollama_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        logger.warning(f"Ollama request failed: {e}")
        return ""


def _score_faithfulness(answer: str, contexts: list[str], ollama_url: str, model: str) -> float:
    """Ask Ollama: is this answer supported by the context? Returns 0.0–1.0."""
    context_text = "\n\n".join(contexts[:3])  # top 3 chunks
    prompt = (
        f"You are an evaluator. Given the context below, rate how faithful the answer is "
        f"to the context on a scale from 0.0 to 1.0, where 1.0 means every claim in the "
        f"answer is directly supported by the context and 0.0 means none are.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Respond with ONLY a single decimal number between 0.0 and 1.0."
    )
    raw = _ask_ollama(prompt, ollama_url, model)
    try:
        score = float(raw.split()[0])
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.5


def _score_answer_relevancy(question: str, answer: str, ollama_url: str, model: str) -> float:
    """Ask Ollama: does this answer address the question? Returns 0.0–1.0."""
    prompt = (
        f"You are an evaluator. Rate how well the answer addresses the question on a "
        f"scale from 0.0 to 1.0, where 1.0 means the answer directly and completely "
        f"answers the question and 0.0 means it is completely irrelevant.\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Respond with ONLY a single decimal number between 0.0 and 1.0."
    )
    raw = _ask_ollama(prompt, ollama_url, model)
    try:
        score = float(raw.split()[0])
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.5


def _score_context_recall(ground_truth: str, contexts: list[str]) -> float:
    """Heuristic: what fraction of ground truth keywords appear in the retrieved context."""
    if not ground_truth:
        return 0.0
    import re
    # Extract meaningful words (4+ chars) from ground truth
    keywords = set(re.findall(r'\b\w{4,}\b', ground_truth.lower()))
    if not keywords:
        return 0.0
    context_text = " ".join(contexts).lower()
    found = sum(1 for kw in keywords if kw in context_text)
    return found / len(keywords)


def _score_context_precision(answer: str, contexts: list[str]) -> float:
    """Heuristic: what fraction of retrieved chunks contain words from the answer."""
    if not contexts or not answer:
        return 0.0
    import re
    answer_words = set(re.findall(r'\b\w{4,}\b', answer.lower()))
    if not answer_words:
        return 0.0
    relevant = sum(
        1 for ctx in contexts
        if any(w in ctx.lower() for w in answer_words)
    )
    return relevant / len(contexts)


def compute_ragas_metrics(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
    ollama_url: str,
    ollama_model: str,
) -> dict[str, float]:
    """Compute RAG quality metrics using Ollama (no OpenAI needed)."""
    logger.info(f"Computing metrics via Ollama ({ollama_model}) for {len(questions)} questions …")

    faithfulness_scores = []
    relevancy_scores = []
    precision_scores = []
    recall_scores = []

    for i, (q, a, ctx, gt) in enumerate(zip(questions, answers, contexts, ground_truths)):
        logger.info(f"  Scoring [{i+1}/{len(questions)}] …")
        faithfulness_scores.append(_score_faithfulness(a, ctx, ollama_url, ollama_model))
        relevancy_scores.append(_score_answer_relevancy(q, a, ollama_url, ollama_model))
        precision_scores.append(_score_context_precision(a, ctx))
        recall_scores.append(_score_context_recall(gt, ctx))

    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    metrics = {
        "faithfulness":       round(mean(faithfulness_scores), 4),
        "answer_relevancy":   round(mean(relevancy_scores), 4),
        "context_precision":  round(mean(precision_scores), 4),
        "context_recall":     round(mean(recall_scores), 4),
    }
    logger.info(f"Metrics: {metrics}")
    return metrics


def save_to_postgres(
    postgres_url: str,
    mlflow_run_id: str,
    pipeline_variant: str,
    metrics: dict,
    num_questions: int,
    params: dict,
    notes: str,
):
    try:
        engine = create_engine(postgres_url, poolclass=NullPool)
        with engine.connect() as conn:
            conn.execute(
                text(
                    """INSERT INTO eval_results
                       (mlflow_run_id, pipeline_variant, faithfulness, answer_relevancy,
                        context_precision, context_recall, num_questions, params, notes)
                       VALUES (:run_id, :variant, :faith, :rel, :prec, :recall, :nq, :params, :notes)
                       ON CONFLICT (mlflow_run_id) DO NOTHING"""
                ),
                {
                    "run_id": mlflow_run_id,
                    "variant": pipeline_variant,
                    "faith": metrics.get("faithfulness"),
                    "rel": metrics.get("answer_relevancy"),
                    "prec": metrics.get("context_precision"),
                    "recall": metrics.get("context_recall"),
                    "nq": num_questions,
                    "params": json.dumps(params),
                    "notes": notes,
                },
            )
            conn.commit()
    except Exception as e:
        logger.warning(f"Postgres save failed: {e}")


async def run_evaluation(
    testset_path: str,
    pipeline_variant: str,
    settings: Settings,
    notes: str = "",
) -> dict[str, Any]:
    """
    Run a full RAGAS evaluation for one pipeline variant.
    Returns dict with metrics + MLflow run_id.
    """
    testset = load_testset(testset_path)
    use_reranker = pipeline_variant == "reranked"

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("arxiv-rag-eval")

    params = {
        "pipeline_variant": pipeline_variant,
        "use_reranker": use_reranker,
        "num_questions": len(testset),
    }

    with mlflow.start_run(run_name=f"eval-{pipeline_variant}-{int(time.time())}") as run:
        mlflow.log_params(params)
        run_id = run.info.run_id
        logger.info(f"MLflow run {run_id} — evaluating '{pipeline_variant}' on {len(testset)} questions")

        questions, answers, contexts, ground_truths = [], [], [], []
        errors = 0

        for i, item in enumerate(testset):
            q = item["question"]
            gt = item.get("ground_truth", "")
            logger.info(f"  [{i+1}/{len(testset)}] {q[:60]}")
            try:
                result = await query_backend(settings.backend_url, q, use_reranker)
                answers.append(result.get("answer", ""))
                ctx = [s.get("excerpt", "") for s in result.get("sources", [])]
                contexts.append(ctx if ctx else [""])
                questions.append(q)
                ground_truths.append(gt)
            except Exception as e:
                logger.warning(f"  Query failed: {e}")
                errors += 1

        mlflow.log_metric("queries_succeeded", len(questions))
        mlflow.log_metric("queries_failed", errors)

        metrics = {}
        if questions:
            metrics = compute_ragas_metrics(
                questions, answers, contexts, ground_truths,
                settings.ollama_url, settings.ollama_model,
            )
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
        else:
            logger.info("No successful queries — skipping metrics")

        save_to_postgres(
            settings.postgres_url,
            run_id,
            pipeline_variant,
            metrics,
            len(questions),
            params,
            notes,
        )

    return {
        "run_id": run_id,
        "pipeline_variant": pipeline_variant,
        "metrics": metrics,
        "num_questions": len(questions),
        "errors": errors,
        "params": params,
    }
