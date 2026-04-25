from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    qdrant_url: str = "http://localhost:6333"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    arxiv_categories: str = "cs.LG,cs.AI,cs.CL"
    papers_per_category: int = 70
    chunk_size: int = 512
    chunk_overlap: int = 64
    collection_name: str = "arxiv_papers"
    cache_dir: str = "/app/cache"

    class Config:
        env_file = ".env"
