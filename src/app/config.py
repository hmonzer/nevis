from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    app_name: str = "Nevis API"
    app_version: str = "1.0.0"
    database_url: str = "postgresql+asyncpg://localhost/nevis"

    # S3 Storage Settings
    s3_bucket_name: str = "nevis-documents"
    s3_endpoint_url: str | None = None  # For LocalStack, set to http://localhost:4566
    aws_region: str = "eu-west-1"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None

    # ML Model Settings
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # For production: BAAI/bge-reranker-v2-m3

    # Search Settings
    search_default_top_k: int = 3
    search_default_threshold: float = 0.3
    search_max_top_k: int = 100
    client_search_default_threshold: float = 0.1

    # Chunking Settings
    chunk_size: int = 300
    chunk_overlap: int = 50

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
