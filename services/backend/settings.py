from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    qdrant_url: str = "http://localhost:6333"
    postgres_url: str = "postgresql://arxiv:arxiv@localhost:5432/arxiv"
    mlflow_tracking_uri: str = "http://localhost:5000"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    collection_name: str = "arxiv_papers"
    openai_api_key: str = ""
    ollama_url: str = "http://host.docker.internal:11434"
    ollama_model: str = "llama3"
    top_k_retrieve: int = 20
    top_k_rerank: int = 5

    class Config:
        env_file = ".env"
