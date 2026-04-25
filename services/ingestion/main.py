"""
ArXiv ingestion service.
- On startup: fetch papers, chunk, embed, upsert to Qdrant
- Daily re-fetch via APScheduler
- FastAPI health + status endpoints
"""
import asyncio
import os
import time
from contextlib import asynccontextmanager

import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel

from ingestion import run_ingestion
from settings import Settings

settings = Settings()
scheduler = AsyncIOScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting ingestion service — seeding Qdrant...")
    await run_ingestion(settings)
    scheduler.add_job(run_ingestion, "interval", hours=24, args=[settings])
    scheduler.start()
    logger.info("Ingestion complete. Scheduler running.")
    yield
    scheduler.shutdown()


app = FastAPI(title="ArXiv Ingestion Service", lifespan=lifespan)

_status = {"last_run": None, "papers_indexed": 0, "status": "idle"}


class StatusResponse(BaseModel):
    last_run: str | None
    papers_indexed: int
    status: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status", response_model=StatusResponse)
def status():
    return _status


@app.post("/ingest")
async def trigger_ingest():
    asyncio.create_task(run_ingestion(settings))
    return {"message": "Ingestion triggered"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, log_level="info")
