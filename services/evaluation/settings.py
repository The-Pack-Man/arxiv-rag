from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    backend_url: str = "http://localhost:8000"
    mlflow_tracking_uri: str = "http://localhost:5000"
    openai_api_key: str = ""  # no longer required
    postgres_url: str = "postgresql://arxiv:arxiv@postgres:5432/arxiv"
    run_on_startup: bool = False
    ollama_url: str = "http://host.docker.internal:11434"
    ollama_model: str = "llama3"

    class Config:
        env_file = ".env"
