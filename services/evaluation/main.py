"""Evaluation service FastAPI app."""
import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException
from loguru import logger
from pydantic import BaseModel

from evaluator import run_evaluation
from settings import Settings

settings = Settings()
_running = False


class EvalRequest(BaseModel):
    testset_path: str = "/app/testset.json"
    pipeline_variant: str = "reranked"
    notes: str = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.run_on_startup:
        logger.info("RUN_ON_STARTUP=true — running evaluation suite...")
        asyncio.create_task(_run_both_variants())
    yield


async def _run_both_variants():
    global _running
    _running = True
    try:
        for variant in ["naive", "reranked"]:
            await run_evaluation("/app/testset.json", variant, settings, notes="auto startup run")
    finally:
        _running = False


app = FastAPI(title="ArXiv RAG Evaluation Service", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "running": _running}


@app.post("/eval/run")
async def start_eval(req: EvalRequest, background_tasks: BackgroundTasks):
    global _running
    if _running:
        raise HTTPException(status_code=409, detail="Evaluation already running")
    background_tasks.add_task(_eval_task, req)
    return {"message": f"Evaluation started for variant '{req.pipeline_variant}'"}


async def _eval_task(req: EvalRequest):
    global _running
    _running = True
    try:
        result = await run_evaluation(
            req.testset_path, req.pipeline_variant, settings, req.notes
        )
        logger.info(f"Eval complete: {result}")
    except Exception as e:
        logger.error(f"Eval failed: {e}")
    finally:
        _running = False


@app.post("/eval/run-both")
async def run_both(background_tasks: BackgroundTasks):
    global _running
    if _running:
        raise HTTPException(status_code=409, detail="Evaluation already running")
    background_tasks.add_task(_run_both_variants)
    return {"message": "Running both naive and reranked evaluations"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8002, log_level="info")
