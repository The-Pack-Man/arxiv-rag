"""Postgres query logging."""
import json
from datetime import datetime

from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool


_engine = None


def get_engine(postgres_url: str):
    global _engine
    if _engine is None:
        _engine = create_engine(postgres_url, poolclass=NullPool)
    return _engine


def log_query(
    postgres_url: str,
    query_text: str,
    answer: str,
    sources: list[dict],
    latency_ms: int,
    model: str,
    top_k: int,
    reranked: bool,
):
    try:
        engine = get_engine(postgres_url)
        with engine.connect() as conn:
            conn.execute(
                text(
                    """INSERT INTO query_logs
                       (query_text, answer, sources, latency_ms, model, top_k, reranked)
                       VALUES (:query_text, :answer, :sources, :latency_ms, :model, :top_k, :reranked)"""
                ),
                {
                    "query_text": query_text,
                    "answer": answer,
                    "sources": json.dumps(sources),
                    "latency_ms": latency_ms,
                    "model": model,
                    "top_k": top_k,
                    "reranked": reranked,
                },
            )
            conn.commit()
    except Exception as e:
        logger.warning(f"Failed to log query: {e}")


def save_eval_result(
    postgres_url: str,
    mlflow_run_id: str,
    pipeline_variant: str,
    metrics: dict,
    num_questions: int,
    params: dict,
    notes: str = "",
):
    try:
        engine = get_engine(postgres_url)
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
        logger.warning(f"Failed to save eval result: {e}")


def get_eval_history(postgres_url: str, limit: int = 50) -> list[dict]:
    try:
        engine = get_engine(postgres_url)
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """SELECT mlflow_run_id, pipeline_variant, faithfulness, answer_relevancy,
                              context_precision, context_recall, num_questions, params, notes,
                              created_at
                       FROM eval_results
                       ORDER BY created_at DESC
                       LIMIT :limit"""
                ),
                {"limit": limit},
            ).fetchall()
            return [dict(r._mapping) for r in rows]
    except Exception as e:
        logger.warning(f"Failed to fetch eval history: {e}")
        return []
