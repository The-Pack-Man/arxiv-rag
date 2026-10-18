# RAG Evaluation

The evaluation service compares two retrieval pipelines on a 25-question ground-truth test set, using [RAGAS](https://docs.ragas.io/) metrics. The judge model runs locally through Ollama by default, so evaluation costs nothing to run.

## Results

Measured 2026-10-18 over 551 indexed papers. The generator was `llama3` and the judge `qwen2.5:7b`, both local via Ollama. 25 questions, one run per variant ([naive](evals/results/20261018T215227Z_naive.json), [reranked](evals/results/20261018T221520Z_reranked.json)). All 200 metric scores were computed; there were no judge failures.

Both variants answered the same 25 questions, so they are compared question by question. Intervals are 95% paired bootstrap ([`scripts/compare_runs.py`](scripts/compare_runs.py)):

| Metric | Naive | Reranked | Difference | 95% interval | Questions better / worse / same |
|---|---|---|---|---|---|
| **Context precision** | 0.30 | **0.50** | **+0.20** | **+0.09 to +0.31** | 13 / 2 / 10 |
| Faithfulness | 0.77 | 0.81 | +0.04 | −0.09 to +0.19 | 9 / 7 / 9 |
| Context recall | 0.57 | 0.58 | +0.01 | −0.18 to +0.19 | 4 / 4 / 17 |
| Answer relevancy | 0.82 | 0.77 | −0.05 | −0.17 to +0.06 | 7 / 10 / 8 |

**Reranking makes retrieval more precise and nothing else measurably.**
- **Context precision:** the only difference whose interval excludes zero. Two thirds more of the chunks sent to the LLM are relevant, and reranking made it worse on only 2 of 25 questions.
- **Recall:** unchanged, as expected. The reranker reorders the same 20 candidates, so it can't recover evidence that dense retrieval missed.
- **Faithfulness and answer relevancy:** the movements are within noise at this sample size.

**Recall is limited by the corpus, not the pipeline.** Ingestion pulls recent papers from three arXiv categories, while many test questions are about classic papers (RoPE, FlashAttention, FAISS, RLHF), which may not be indexed. Four questions have recall below 0.3 under both variants.

**When reranking hurts, it's precision without coverage.** Asked about the KL term in the RLHF objective, the reranker promoted a chunk about the KL term in a VAE objective. That's the best lexical match in an index that may not contain the RLHF paper, and the answer followed it. Asked how BERT and GPT differ, the reranked context made the model say the sources don't compare them directly. That's arguably more honest, but it scores 0 on relevancy.

**An earlier version of this file was wrong.** It reported answer relevancy +19% from reranking. Those numbers were never produced by this code and were removed. The measured effect on relevancy is −0.05, within noise.

**Limits:** 25 questions, one run, a 7B judge. Compare variants only under the same judge, and treat differences under about 0.1 as noise unless the interval says otherwise.

## Pipeline variants

| Variant | Retrieval | Reranker | Contexts sent to the LLM |
|---|---|---|---|
| **naive** | Qdrant cosine ANN (`all-MiniLM-L6-v2`), top-20 | none | first 5 of the top 20 |
| **reranked** | same top-20 candidates | `cross-encoder/ms-marco-MiniLM-L-6-v2` | top 5 after reranking |

Both variants go through the same backend `/chat` endpoint. Only the `use_reranker` flag differs.

## Metrics

All four are RAGAS LLM-judged metrics, scored 0–1 (higher is better).

| Metric | What it measures | Inputs |
|---|---|---|
| **Faithfulness** | Share of the answer's claims that the retrieved context supports. Measures hallucination. | question, answer, contexts |
| **Answer relevancy** | How directly the answer addresses the question. The judge generates questions from the answer, which are compared to the original by embedding similarity. | question, answer |
| **Context precision** | Whether the relevant chunks are ranked near the top, judged against the reference answer. | question, reference, contexts |
| **Context recall** | Share of the reference answer that the retrieved context supports. | question, reference, contexts |

The contexts scored are the **full chunk texts the LLM received**, returned by `/chat` as `contexts`. They are not the shortened excerpts shown in the UI.

## Judge configuration

| Setting | Default | Notes |
|---|---|---|
| `JUDGE_MODEL` | `qwen2.5:7b` | Deliberately a different model family from the default generator (`llama3`), to avoid a model grading its own answers |
| `JUDGE_BASE_URL` | `{OLLAMA_URL}/v1` | Any OpenAI-compatible endpoint works (Ollama, LM Studio, vLLM, hosted APIs) |
| `JUDGE_API_KEY` | empty | Only needed for hosted endpoints |
| `JUDGE_CONCURRENCY` | `2` | Parallel questions being scored. Keep this low for a local GPU |
| Embeddings | `all-MiniLM-L6-v2` (local) | Used by answer relevancy only |

Things to keep in mind when reading results:

- **A 7B judge is noisier than a frontier model.** Compare variants scored by the same judge, and do not compare absolute numbers across judges. For a stronger free option, try a larger local model such as `qwen2.5:14b` if your hardware allows.
- **Failures are reported, not hidden.** If the judge returns invalid output, that metric is recorded as `null` for that question and counted in `<metric>_failed`. Means are taken only over questions that succeeded. Check the failure counts before trusting a mean.
- **25 questions is a small sample.** Differences of a few points between variants may be noise. Run each variant more than once to see the variance.

## How to run

```bash
# 1. Pull the judge model (and the generator, if you're not using OpenAI)
ollama pull qwen2.5:7b
ollama pull llama3

# 2. Start the stack and wait for ingestion
docker compose up --build

# 3. Score both variants
docker compose exec evaluation curl -s -X POST localhost:8002/eval/run-both

# 4. Follow progress
docker compose logs -f evaluation
```

Each run writes `evals/results/<timestamp>_<variant>.json`, logs to MLflow (http://localhost:5001, experiment `arxiv-rag-eval`), and appears on the dashboard at http://arxiv.localhost/eval.

Commit the JSON files you want to report, then summarize them in the Results section above.

## Result file format

```json
{
  "run_id": "…",
  "pipeline_variant": "reranked",
  "generator_model": "ollama:llama3",
  "params": { "judge_model": "qwen2.5:7b", "judge_base_url": "…", "embedding_model": "…" },
  "summary": { "faithfulness": 0.0, "faithfulness_n": 0, "faithfulness_failed": 0, "…": "…" },
  "query_failures": [],
  "rows": [ { "id": 1, "user_input": "…", "response": "…", "retrieved_contexts": ["…"],
              "scores": { "faithfulness": 0.0, "…": "…" }, "errors": {} } ]
}
```

## Comparing two runs

```bash
python scripts/compare_runs.py evals/results/<naive>.json evals/results/<reranked>.json
```

## Tests

```bash
cd services/evaluation
pip install -r requirements.txt pytest pytest-asyncio
pytest
```

The suite includes an integration test that runs the real RAGAS metrics against a local fake OpenAI-compatible server. That exercises the same request path Ollama uses, without needing a model.

## Test set coverage

Counts come from the `category` field in `services/evaluation/testset.json`.

| Category | Count |
|---|---:|
| alignment | 4 |
| architecture | 4 |
| inference | 3 |
| language_models | 2 |
| training | 2 |
| transformers, fine_tuning, generative_models, prompting, rag, multimodal, interpretability, retrieval, reinforcement_learning, compression | 1 each |
| **Total** | **25** |

## Known failure modes to watch for

1. **Multi-hop questions.** Questions that need synthesis across papers (e.g. "compare RLHF vs DPO") may retrieve chunks from only one paper.
2. **Freshness.** Recent papers are only covered after ingestion re-runs.
3. **Exact numbers.** Questions about specific experimental results are hard, because chunks rarely keep the full numerical context.
