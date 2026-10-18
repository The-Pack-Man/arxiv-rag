"""
RAGAS metric construction.

The judge is any OpenAI-compatible chat endpoint. The default is a local
Ollama server, so scoring is free. LM Studio, vLLM, a hosted free tier, or
OpenAI work too by changing JUDGE_BASE_URL / JUDGE_MODEL / JUDGE_API_KEY.

Answer relevancy also needs embeddings. These run locally with
sentence-transformers (the same all-MiniLM-L6-v2 the retriever uses).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from settings import Settings

# Metric name -> which sample fields its ascore() takes.
METRIC_INPUTS: dict[str, tuple[str, ...]] = {
    "faithfulness": ("user_input", "response", "retrieved_contexts"),
    "answer_relevancy": ("user_input", "response"),
    "context_precision": ("user_input", "reference", "retrieved_contexts"),
    "context_recall": ("user_input", "retrieved_contexts", "reference"),
}


@dataclass
class JudgeConfig:
    base_url: str
    model: str
    embedding_model: str
    max_tokens: int

    def as_params(self) -> dict[str, Any]:
        return {
            "judge_base_url": self.base_url,
            "judge_model": self.model,
            "embedding_model": self.embedding_model,
        }


def judge_config(settings: Settings) -> JudgeConfig:
    base_url = settings.judge_base_url or settings.ollama_url.rstrip("/") + "/v1"
    return JudgeConfig(
        base_url=base_url,
        model=settings.judge_model,
        embedding_model=settings.embedding_model,
        max_tokens=settings.judge_max_tokens,
    )


def build_metrics(settings: Settings) -> dict[str, Any]:
    """Construct the four RAGAS metrics against the configured judge."""
    from openai import AsyncOpenAI
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithReference,
        ContextRecall,
        Faithfulness,
    )

    cfg = judge_config(settings)
    client = AsyncOpenAI(base_url=cfg.base_url, api_key=settings.judge_api_key or "ollama")
    llm = llm_factory(cfg.model, provider="openai", client=client, max_tokens=cfg.max_tokens)
    embeddings = HuggingFaceEmbeddings(model=cfg.embedding_model)

    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecisionWithReference(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }


MetricsBuilder = Callable[[Settings], dict[str, Any]]
