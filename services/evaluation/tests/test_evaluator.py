import json
import math

import pytest

import evaluator
import judge
from conftest import FakeEmbeddings
from settings import Settings

SAMPLE = {
    "id": 1,
    "category": "transformers",
    "user_input": "What is attention?",
    "reference": "Attention computes a weighted sum of values.",
    "response": "Attention weights values by query-key similarity [Source 1].",
    "retrieved_contexts": ["Attention computes a weighted sum of value vectors."],
}


class FixedMetric:
    def __init__(self, value):
        self.value = value

    async def ascore(self, **kwargs):
        if isinstance(self.value, Exception):
            raise self.value
        return type("R", (), {"value": self.value})()


def fixed_metrics(**values):
    return {name: FixedMetric(values.get(name, 0.5)) for name in evaluator.METRIC_NAMES}


@pytest.mark.asyncio
async def test_failed_metric_is_null_not_default():
    metrics = fixed_metrics(faithfulness=RuntimeError("judge returned invalid JSON"))
    result = await evaluator.score_sample(metrics, SAMPLE)
    assert result["scores"]["faithfulness"] is None
    assert "RuntimeError" in result["errors"]["faithfulness"]
    assert result["scores"]["context_recall"] == 0.5


@pytest.mark.asyncio
async def test_nan_score_counts_as_failure():
    result = await evaluator.score_sample(fixed_metrics(answer_relevancy=math.nan), SAMPLE)
    assert result["scores"]["answer_relevancy"] is None
    assert "NaN" in result["errors"]["answer_relevancy"]


def test_aggregate_excludes_failures_and_counts_them():
    rows = [
        {"scores": {"faithfulness": 1.0, "answer_relevancy": 0.5, "context_precision": None, "context_recall": 0.0}},
        {"scores": {"faithfulness": 0.0, "answer_relevancy": None, "context_precision": None, "context_recall": 1.0}},
    ]
    out = evaluator.aggregate(rows)
    assert out["faithfulness"] == 0.5 and out["faithfulness_failed"] == 0
    assert out["answer_relevancy"] == 0.5 and out["answer_relevancy_n"] == 1
    assert out["context_precision"] is None and out["context_precision_failed"] == 2


def test_judge_defaults_to_local_ollama():
    cfg = judge.judge_config(Settings(ollama_url="http://host.docker.internal:11434/"))
    assert cfg.base_url == "http://host.docker.internal:11434/v1"
    assert cfg.model == "qwen2.5:7b"


@pytest.mark.asyncio
async def test_run_evaluation_writes_artifact(tmp_path, monkeypatch):
    testset = tmp_path / "testset.json"
    testset.write_text(json.dumps([
        {"id": 1, "question": "Q1", "ground_truth": "A1", "category": "c"},
        {"id": 2, "question": "Q2", "ground_truth": "A2", "category": "c"},
    ]))

    async def fake_backend(url, question, use_reranker, timeout=180):
        if question == "Q2":
            raise RuntimeError("backend down")
        return {"answer": "ans", "contexts": ["full chunk text"], "model": "ollama:llama3"}

    monkeypatch.setattr(evaluator, "query_backend", fake_backend)
    settings = Settings(
        mlflow_tracking_uri=f"file://{tmp_path}/mlruns",
        results_dir=str(tmp_path / "results"),
        postgres_url="postgresql://nobody@127.0.0.1:1/none",
    )
    out = await evaluator.run_evaluation(
        str(testset), "reranked", settings, metrics_builder=lambda s: fixed_metrics(faithfulness=0.8)
    )

    assert out["num_questions"] == 1 and out["errors"] == 1
    payload = json.loads(open(out["artifact"]).read())
    assert payload["summary"]["faithfulness"] == 0.8
    assert payload["generator_model"] == "ollama:llama3"
    assert payload["params"]["judge_model"] == "qwen2.5:7b"
    assert payload["rows"][0]["retrieved_contexts"] == ["full chunk text"]
    assert payload["query_failures"][0]["id"] == 2


@pytest.mark.asyncio
async def test_real_ragas_metrics_against_openai_compatible_judge(fake_judge, monkeypatch):
    """Runs the actual RAGAS metric code through an Ollama-style /v1 endpoint."""
    monkeypatch.setattr("ragas.embeddings.HuggingFaceEmbeddings", lambda model: FakeEmbeddings())
    host, port = fake_judge.server_address
    settings = Settings(judge_base_url=f"http://{host}:{port}/v1", judge_model="qwen2.5:7b")

    metrics = judge.build_metrics(settings)
    result = await evaluator.score_sample(metrics, SAMPLE)

    assert result["errors"] == {}, result["errors"]
    for name in evaluator.METRIC_NAMES:
        assert 0.0 <= result["scores"][name] <= 1.0
    assert fake_judge.requests and all(r["model"] == "qwen2.5:7b" for r in fake_judge.requests)
    # JSON mode, which Ollama's OpenAI-compatible endpoint supports
    assert all(r.get("response_format", {}).get("type") == "json_object" for r in fake_judge.requests)
