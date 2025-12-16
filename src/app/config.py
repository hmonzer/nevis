import logging
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


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
    search_default_threshold: float = 0.5
    search_max_top_k: int = 100
    client_search_default_threshold: float = 0.1

    # Chunking Settings (token-based, using embedding model's tokenizer)
    chunk_size: int = 256  # Maximum tokens per chunk
    chunk_overlap: int = 25  # ~10% overlap in tokens

    # Summarization Settings
    summarization_enabled: bool = True  # Enable/disable summarization feature
    summarization_provider: str = "claude"  # Options: "claude", "gemini"
    anthropic_api_key: str | None = None  # Anthropic API key for Claude
    google_api_key: str | None = None  # Google API key for Gemini
    claude_model: str = "claude-sonnet-4-20250514"  # Claude model for summarization
    gemini_model: str = "models/gemini-flash-latest"  # Gemini model for summarization

    model_config = SettingsConfigDict(
        # pydantic-settings reads from environment variables first, then .env file
        # In Docker, env vars are injected by docker-compose from .env file
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def summarization_available(self) -> bool:
        """Check if summarization is enabled and has required API key."""
        if not self.summarization_enabled:
            return False
        if self.summarization_provider == "claude" and self.anthropic_api_key:
            return True
        if self.summarization_provider == "gemini" and self.google_api_key:
            return True
        return False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
