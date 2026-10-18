"""
RAGAS evaluation runner.

Runs the ground-truth test set through the backend for one pipeline variant
(naive or reranked), scores every answer with four RAGAS metrics, and records
the results in three places:

- a JSON artifact in /app/results (mounted to evals/results/ in the repo),
- MLflow (params, aggregate metrics, and the JSON as an artifact),
- Postgres eval_results (for the frontend dashboard).

A metric that fails for a question is recorded as null and counted. It is
never replaced with a default score, so judge failures stay visible.
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import mlflow
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

from judge import METRIC_INPUTS, MetricsBuilder, build_metrics, judge_config
from settings import Settings

METRIC_NAMES = list(METRIC_INPUTS)


def load_testset(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


async def query_backend(
    backend_url: str,
    question: str,
    use_reranker: bool,
    timeout: int = 180,
) -> dict[str, Any]:
    """Call the backend /chat endpoint."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{backend_url}/chat",
            json={"query": question, "use_reranker": use_reranker},
        )
        resp.raise_for_status()
        return resp.json()


async def score_sample(metrics: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    """Score one sample with every metric. Failures become None plus an error message."""
    scores: dict[str, float | None] = {}
    errors: dict[str, str] = {}
    for name, metric in metrics.items():
        kwargs = {field: sample[field] for field in METRIC_INPUTS[name]}
        try:
            result = await metric.ascore(**kwargs)
            value = float(getattr(result, "value", result))
            if math.isnan(value):
                raise ValueError("judge returned NaN")
            scores[name] = value
        except Exception as e:  # judge errors must not be hidden
            scores[name] = None
            errors[name] = f"{type(e).__name__}: {e}"[:300]
    return {"scores": scores, "errors": errors}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean of each metric over questions where it succeeded, plus failure counts."""
    out: dict[str, Any] = {}
    for name in METRIC_NAMES:
        values = [r["scores"][name] for r in rows if r["scores"].get(name) is not None]
        out[name] = round(sum(values) / len(values), 4) if values else None
        out[f"{name}_n"] = len(values)
        out[f"{name}_failed"] = len(rows) - len(values)
    return out


async def score_all(
    metrics: dict[str, Any], samples: list[dict[str, Any]], concurrency: int
) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(max(1, concurrency))

    async def one(i: int, s: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            logger.info(f"  scoring [{i + 1}/{len(samples)}] {s['user_input'][:60]}")
            return {**s, **await score_sample(metrics, s)}

    return await asyncio.gather(*(one(i, s) for i, s in enumerate(samples)))


def write_artifact(results_dir: str, payload: dict[str, Any]) -> Path:
    path = Path(results_dir)
    path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = path / f"{stamp}_{payload['pipeline_variant']}.json"
    out.write_text(json.dumps(payload, indent=2))
    return out


def save_to_postgres(
    postgres_url: str,
    mlflow_run_id: str,
    pipeline_variant: str,
    metrics: dict,
    num_questions: int,
    params: dict,
    notes: str,
) -> None:
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
    metrics_builder: MetricsBuilder = build_metrics,
) -> dict[str, Any]:
    """Run the full evaluation for one pipeline variant."""
    testset = load_testset(testset_path)
    use_reranker = pipeline_variant == "reranked"
    judge = judge_config(settings)
    metrics = metrics_builder(settings)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("arxiv-rag-eval")

    params = {
        "pipeline_variant": pipeline_variant,
        "use_reranker": use_reranker,
        "num_questions": len(testset),
        "metrics": ",".join(METRIC_NAMES),
        **judge.as_params(),
    }

    with mlflow.start_run(run_name=f"eval-{pipeline_variant}-{int(time.time())}") as run:
        mlflow.log_params(params)
        run_id = run.info.run_id
        logger.info(f"MLflow run {run_id}: '{pipeline_variant}' on {len(testset)} questions, judge {judge.model}")

        samples: list[dict[str, Any]] = []
        query_failures: list[dict[str, str]] = []
        generator_model = None
        for i, item in enumerate(testset):
            q = item["question"]
            logger.info(f"  querying [{i + 1}/{len(testset)}] {q[:60]}")
            try:
                result = await query_backend(settings.backend_url, q, use_reranker)
            except Exception as e:
                logger.warning(f"  query failed: {e}")
                query_failures.append({"id": item.get("id"), "error": str(e)[:300]})
                continue
            generator_model = result.get("model", generator_model)
            samples.append(
                {
                    "id": item.get("id"),
                    "category": item.get("category"),
                    "user_input": q,
                    "reference": item.get("ground_truth", ""),
                    "response": result.get("answer", ""),
                    "retrieved_contexts": result.get("contexts") or [""],
                }
            )

        rows = await score_all(metrics, samples, settings.judge_concurrency) if samples else []
        summary = aggregate(rows) if rows else {}

        mlflow.log_param("generator_model", generator_model or "unknown")
        mlflow.log_metric("queries_succeeded", len(samples))
        mlflow.log_metric("queries_failed", len(query_failures))
        for k, v in summary.items():
            if v is not None:
                mlflow.log_metric(k, v)

        payload = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_variant": pipeline_variant,
            "generator_model": generator_model,
            "params": params,
            "summary": summary,
            "query_failures": query_failures,
            "rows": rows,
            "notes": notes,
        }
        artifact = write_artifact(settings.results_dir, payload)
        mlflow.log_artifact(str(artifact))
        logger.info(f"Summary: {summary}")
        logger.info(f"Wrote {artifact}")

        save_to_postgres(
            settings.postgres_url, run_id, pipeline_variant, summary, len(samples), params, notes
        )

    return {
        "run_id": run_id,
        "pipeline_variant": pipeline_variant,
        "metrics": summary,
        "num_questions": len(samples),
        "errors": len(query_failures),
        "artifact": str(artifact),
        "params": params,
    }
