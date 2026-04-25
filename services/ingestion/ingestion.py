"""Core ingestion logic: fetch → parse → chunk → embed → upsert."""
import hashlib
import io
import os
import time
from pathlib import Path
from typing import Iterator

import arxiv
import fitz  # PyMuPDF
import httpx
import tiktoken
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential

from settings import Settings

_tokenizer = tiktoken.get_encoding("cl100k_base")
_embedder: SentenceTransformer | None = None


def get_embedder(model_name: str) -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info(f"Loading embedding model: {model_name}")
        _embedder = SentenceTransformer(model_name)
    return _embedder


def tokenize(text: str) -> list[int]:
    return _tokenizer.encode(text)


def decode_tokens(tokens: list[int]) -> str:
    return _tokenizer.decode(tokens)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Chunk text by token count with overlap."""
    tokens = tokenize(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = decode_tokens(chunk_tokens)
        if chunk_text.strip():
            chunks.append(chunk_text.strip())
        if end >= len(tokens):
            break
        start += chunk_size - overlap
    return chunks


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def download_pdf(url: str, cache_dir: str) -> bytes | None:
    """Download PDF with caching."""
    url_hash = hashlib.md5(url.encode()).hexdigest()
    cache_path = Path(cache_dir) / f"{url_hash}.pdf"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        return cache_path.read_bytes()

    try:
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.content
            cache_path.write_bytes(data)
            return data
    except Exception as e:
        logger.warning(f"PDF download failed for {url}: {e}")
        return None


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = []
        for page in doc:
            pages.append(page.get_text())
        return "\n".join(pages)
    except Exception as e:
        logger.warning(f"PDF text extraction failed: {e}")
        return ""


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int):
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        logger.info(f"Creating Qdrant collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def get_existing_ids(client: QdrantClient, collection_name: str) -> set[str]:
    """Scroll all point IDs from collection for dedup."""
    existing = set()
    offset = None
    while True:
        results, offset = client.scroll(
            collection_name=collection_name,
            limit=1000,
            offset=offset,
            with_payload=["arxiv_id"],
            with_vectors=False,
        )
        for pt in results:
            if pt.payload and "arxiv_id" in pt.payload:
                existing.add(pt.payload["arxiv_id"])
        if offset is None:
            break
    return existing


async def run_ingestion(settings: Settings):
    """Main ingestion pipeline."""
    from main import _status

    _status["status"] = "running"
    start = time.time()

    logger.info("Initializing Qdrant client...")
    client = QdrantClient(url=settings.qdrant_url, timeout=60)

    embedder = get_embedder(settings.embedding_model)
    vector_size = embedder.get_sentence_embedding_dimension()

    ensure_collection(client, settings.collection_name, vector_size)
    existing_ids = get_existing_ids(client, settings.collection_name)
    logger.info(f"Existing papers in Qdrant: {len(existing_ids)}")

    categories = [c.strip() for c in settings.arxiv_categories.split(",")]
    total_indexed = 0

    for category in categories:
        logger.info(f"Fetching papers for category: {category}")
        try:
            papers = list(
                arxiv.Client().results(
                    arxiv.Search(
                        query=f"cat:{category}",
                        max_results=settings.papers_per_category,
                        sort_by=arxiv.SortCriterion.SubmittedDate,
                    )
                )
            )
        except Exception as e:
            logger.error(f"ArXiv API error for {category}: {e}")
            continue

        for paper in papers:
            arxiv_id = paper.entry_id.split("/")[-1]
            if arxiv_id in existing_ids:
                continue

            logger.info(f"Processing {arxiv_id}: {paper.title[:60]}")

            # Try PDF first, fall back to abstract-only
            text = ""
            pdf_url = paper.pdf_url
            if pdf_url:
                pdf_bytes = download_pdf(pdf_url, settings.cache_dir)
                if pdf_bytes:
                    text = extract_text_from_pdf(pdf_bytes)

            if not text:
                text = f"{paper.title}\n\n{paper.summary}"

            chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
            if not chunks:
                continue

            # Embed all chunks
            embeddings = embedder.encode(chunks, show_progress_bar=False)

            # Build Qdrant points
            points = []
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                point_id_str = f"{arxiv_id}_{i}"
                point_id = int(hashlib.md5(point_id_str.encode()).hexdigest(), 16) % (2**63)
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=emb.tolist(),
                        payload={
                            "arxiv_id": arxiv_id,
                            "title": paper.title,
                            "authors": [a.name for a in paper.authors[:10]],
                            "abstract": paper.summary[:1000],
                            "url": paper.entry_id,
                            "pdf_url": pdf_url or "",
                            "published": paper.published.isoformat() if paper.published else "",
                            "categories": [c for c in paper.categories],
                            "chunk_index": i,
                            "chunk_text": chunk,
                        },
                    )
                )

            if points:
                client.upsert(collection_name=settings.collection_name, points=points)
                existing_ids.add(arxiv_id)
                total_indexed += 1
                logger.info(f"  Indexed {len(points)} chunks for {arxiv_id}")

        # Brief pause between categories to be polite to ArXiv API
        time.sleep(3)

    elapsed = time.time() - start
    _status.update(
        {
            "last_run": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "papers_indexed": len(existing_ids),
            "status": "idle",
        }
    )
    logger.info(f"Ingestion complete. {total_indexed} new papers indexed in {elapsed:.1f}s. Total: {len(existing_ids)}")
