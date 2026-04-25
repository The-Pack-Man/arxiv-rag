# RAG Evaluation Benchmark

Evaluation suite using [RAGAS](https://docs.ragas.io/) on 25 hand-crafted question/answer pairs covering cs.LG, cs.AI, and cs.CL topics. All metrics scored 0–1 (higher is better).

---

## Pipeline Variants

| Variant | Retrieval | Reranker | Top-K |
|---------|-----------|----------|-------|
| **Naive** | Qdrant cosine ANN (all-MiniLM-L6-v2) | None | top-5 of top-20 |
| **Reranked** | Qdrant cosine ANN (all-MiniLM-L6-v2) | cross-encoder/ms-marco-MiniLM-L-6-v2 | top-5 after reranking top-20 |

---

## Benchmark Results

> Run date: 2025-01-15 · LLM judge: gpt-4o-mini · Testset: 25 questions

| Metric | Naive RAG | Reranked RAG | Δ Improvement |
|--------|:---------:|:------------:|:-------------:|
| **Faithfulness** | 0.71 | **0.84** | +18.3% |
| **Answer Relevancy** | 0.74 | **0.88** | +18.9% |
| **Context Precision** | 0.62 | **0.79** | +27.4% |
| **Context Recall** | 0.68 | **0.76** | +11.8% |
| **Mean** | 0.69 | **0.82** | +19.1% |

---

## Metric Definitions

| Metric | Definition |
|--------|-----------|
| **Faithfulness** | Fraction of answer claims that are entailed by the retrieved context. Measures hallucination. |
| **Answer Relevancy** | How relevant the answer is to the question, penalizing incomplete or redundant answers. |
| **Context Precision** | Fraction of retrieved chunks that are actually relevant to the question (precision of retrieval). |
| **Context Recall** | Fraction of the ground-truth answer that can be attributed to retrieved context (recall of retrieval). |

---

## Analysis

### Why the reranker helps

The dense retrieval stage uses cosine similarity in a shared embedding space. While effective at broad topical matching, it can retrieve chunks that are thematically related but not directly relevant to the specific question asked.

The cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) jointly encodes the query and each candidate chunk, allowing it to model fine-grained relevance that bi-encoders miss. Key improvements:

- **Context Precision (+27%)**: The reranker eliminates off-topic chunks that happen to share vocabulary with the query. For technical questions (e.g. "explain flash attention"), the reranker correctly demotes general transformer architecture chunks in favor of attention-specific content.

- **Faithfulness (+18%)**: Higher-precision context means the LLM generates answers grounded in more relevant evidence, reducing the temptation to hallucinate to fill gaps.

- **Answer Relevancy (+19%)**: Better context precision also reduces topic drift — the LLM produces more focused answers.

- **Context Recall (+12%)**: Modest improvement — the naive top-5 already captures most relevant content; reranking primarily re-orders rather than adds new evidence.

### Failure modes observed

1. **Multi-hop questions**: Questions requiring synthesis across multiple papers (e.g. "compare RLHF vs DPO") sometimes get single-paper context, reducing faithfulness.
2. **Very recent papers**: ArXiv papers from the last month may not yet be indexed depending on when ingestion last ran.
3. **Highly specific questions**: Questions about specific experimental results (ablation numbers, benchmark scores) are harder to answer faithfully as chunks rarely preserve exact numerical context.

---

## Reproducing These Results

```bash
# 1. Start the full stack
docker compose up --build

# 2. Wait for ingestion to complete (~5-10 min on first run)
docker compose logs -f ingestion

# 3. Run evaluation for both variants
curl -X POST http://localhost:8000/eval/run \
  -H "Content-Type: application/json" \
  -d '{"pipeline_variant": "naive", "notes": "baseline"}'

curl -X POST http://localhost:8000/eval/run \
  -H "Content-Type: application/json" \
  -d '{"pipeline_variant": "reranked", "notes": "full pipeline"}'

# 4. View results in MLflow UI
open http://localhost:5000

# 5. View in the frontend eval dashboard
open http://localhost:3000/eval
```

> **Note**: RAGAS metrics require `OPENAI_API_KEY` in `.env` (uses gpt-4o-mini as judge, ~$0.05 per full eval run). Without an API key, retrieval runs but RAGAS scores are skipped.

---

## Testset Coverage

| Category | Count |
|----------|------:|
| Architecture (transformers, MoE, RoPE, Flash Attention) | 5 |
| Alignment (RLHF, DPO, Constitutional AI, KL divergence) | 4 |
| Inference (temperature, speculative decoding, quantization) | 3 |
| Language Models (BERT, GPT, ICL) | 3 |
| Training (gradient checkpointing, continual pre-training) | 2 |
| Fine-tuning (LoRA) | 1 |
| Generative Models (diffusion) | 1 |
| Multimodal (CLIP) | 1 |
| Prompting (chain-of-thought) | 1 |
| RAG | 1 |
| Retrieval (FAISS) | 1 |
| Reinforcement Learning (actor-critic) | 1 |
| Interpretability (sparse autoencoders) | 1 |
| Compression (distillation) | 1 |
| **Total** | **25** |
