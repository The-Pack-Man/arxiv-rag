from pydantic import BaseModel
from typing import Any


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


class PaperResult(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: str
    published: str
    score: float
    chunk_text: str


class SearchResponse(BaseModel):
    results: list[PaperResult]
    query: str
    total: int


class ChatRequest(BaseModel):
    query: str
    top_k: int = 5
    use_reranker: bool = True


class SourceCitation(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    url: str
    excerpt: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    model: str
    latency_ms: int
    reranked: bool
    contexts: list[str] = []  # full chunk texts given to the LLM (used by evaluation)


class EvalRunRequest(BaseModel):
    testset_path: str = "/app/testset.json"
    pipeline_variant: str = "reranked"  # 'naive' | 'reranked'
    notes: str = ""


class EvalResult(BaseModel):
    run_id: str
    pipeline_variant: str
    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    context_recall: float | None
    num_questions: int
    params: dict[str, Any]
    created_at: str
    notes: str
