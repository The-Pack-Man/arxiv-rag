from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    backend_url: str = "http://localhost:8000"
    mlflow_tracking_uri: str = "http://localhost:5000"
    postgres_url: str = "postgresql://arxiv:arxiv@postgres:5432/arxiv"
    run_on_startup: bool = False
    results_dir: str = "/app/results"

    # Judge: any OpenAI-compatible endpoint. Defaults to local Ollama (free).
    ollama_url: str = "http://host.docker.internal:11434"
    judge_base_url: str = ""          # empty -> {ollama_url}/v1
    judge_model: str = "qwen2.5:7b"
    judge_api_key: str = ""           # Ollama ignores it; set for hosted endpoints
    judge_max_tokens: int = 4096
    judge_concurrency: int = 2

    # Local embeddings for answer relevancy
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
