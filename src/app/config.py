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
    aws_region: str = "us-east-1"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
