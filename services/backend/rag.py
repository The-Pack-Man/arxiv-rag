"""RAG pipeline: embed → retrieve → rerank → generate."""
import time
from typing import Any

import httpx
from loguru import logger
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from settings import Settings

_embedder: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None


def get_embedder(model_name: str) -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info(f"Loading embedder: {model_name}")
        _embedder = SentenceTransformer(model_name)
    return _embedder


def get_reranker(model_name: str) -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info(f"Loading reranker: {model_name}")
        _reranker = CrossEncoder(model_name)
    return _reranker


def embed_query(query: str, model_name: str) -> list[float]:
    emb = get_embedder(model_name)
    return emb.encode(query).tolist()


def retrieve(
    query: str,
    client: QdrantClient,
    settings: Settings,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Semantic search against Qdrant."""
    k = top_k or settings.top_k_retrieve
    vec = embed_query(query, settings.embedding_model)
    hits = client.search(
        collection_name=settings.collection_name,
        query_vector=vec,
        limit=k,
        with_payload=True,
    )
    return [
        {
            "score": h.score,
            "payload": h.payload,
        }
        for h in hits
    ]


def rerank(query: str, candidates: list[dict[str, Any]], model_name: str, top_k: int) -> list[dict[str, Any]]:
    """Cross-encoder reranking."""
    reranker = get_reranker(model_name)
    pairs = [(query, c["payload"]["chunk_text"]) for c in candidates]
    scores = reranker.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]


def build_prompt(query: str, contexts: list[dict[str, Any]]) -> str:
    ctx_blocks = []
    for i, c in enumerate(contexts, 1):
        p = c["payload"]
        ctx_blocks.append(
            f"[Source {i}] {p['title']} (arxiv:{p['arxiv_id']})\n{p['chunk_text']}"
        )
    context_str = "\n\n---\n\n".join(ctx_blocks)
    return f"""You are a research assistant specializing in machine learning papers. Answer the question using ONLY the provided context. Cite sources by their [Source N] tag.

CONTEXT:
{context_str}

QUESTION: {query}

ANSWER (cite sources inline, e.g. [Source 1]):"""


async def call_llm(prompt: str, settings: Settings) -> tuple[str, str]:
    """Call OpenAI if key set, else Ollama. Returns (answer, model_name)."""
    if settings.openai_api_key:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024,
        )
        return resp.choices[0].message.content, "gpt-4o-mini"
    else:
        # Ollama fallback
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{settings.ollama_url}/api/generate",
                json={
                    "model": settings.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 1024},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["response"], f"ollama:{settings.ollama_model}"


async def run_rag_pipeline(
    query: str,
    qdrant_client: QdrantClient,
    settings: Settings,
    use_reranker: bool = True,
    top_k_override: int | None = None,
) -> dict[str, Any]:
    """Full RAG pipeline. Returns answer + sources + metadata."""
    t0 = time.monotonic()

    # Retrieve
    candidates = retrieve(query, qdrant_client, settings, top_k=settings.top_k_retrieve)

    # Rerank or truncate
    if use_reranker and candidates:
        top = rerank(query, candidates, settings.reranker_model, settings.top_k_rerank)
    else:
        top = candidates[: (top_k_override or settings.top_k_rerank)]

    # Generate
    prompt = build_prompt(query, top)
    try:
        answer, model = await call_llm(prompt, settings)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        answer = f"[LLM unavailable: {e}]\n\nRelevant papers found:\n" + "\n".join(
            f"- {c['payload']['title']}" for c in top
        )
        model = "unavailable"

    latency_ms = int((time.monotonic() - t0) * 1000)

    sources = []
    seen_ids = set()
    for c in top:
        p = c["payload"]
        aid = p.get("arxiv_id", "")
        if aid not in seen_ids:
            seen_ids.add(aid)
            sources.append(
                {
                    "arxiv_id": aid,
                    "title": p.get("title", ""),
                    "authors": p.get("authors", []),
                    "url": p.get("url", ""),
                    "excerpt": p.get("chunk_text", "")[:300],
                    "score": c.get("rerank_score", c["score"]),
                }
            )

    return {
        "answer": answer,
        "sources": sources,
        "model": model,
        "latency_ms": latency_ms,
        "reranked": use_reranker,
        "contexts": [c["payload"]["chunk_text"] for c in top],
    }
