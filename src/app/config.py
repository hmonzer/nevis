"""Application configuration with structured settings groups."""
import logging
from functools import lru_cache

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


# =============================================================================
# Nested Settings Models
# =============================================================================


class SearchSettings(BaseModel):
    """Search pagination and general settings."""

    default_top_k: int = 3
    max_top_k: int = 100


class ClientSearchSettings(BaseModel):
    """
    Client search settings using PostgreSQL pg_trgm.

    Trigram similarity scores range from 0 to 1.
    Higher threshold = stricter matching.
    retrieval_multiplier: How many extra candidates to fetch when reranking is enabled.
    reranker_score_threshold: Minimum cross-encoder score for client results.
        Client descriptions are typically short, so they may score lower than documents.
    """

    trgm_threshold: float = 0.32
    retrieval_multiplier: int = 3
    reranker_score_threshold: float = 1.5


class ChunkSearchSettings(BaseModel):
    """
    Document chunk search settings for hybrid vector + keyword search.

    Vector similarity uses cosine similarity ranging from -1 to 1.
    Retrieval multipliers control how many candidates to fetch before ranking.
    reranker_score_threshold: Minimum cross-encoder score for chunk results.
        Document chunks typically have more content and score higher than client descriptions.
    """

    vector_similarity_threshold: float = 0.3
    retrieval_multiplier_with_rerank: int = 3
    retrieval_multiplier_no_rerank: int = 2
    reranker_score_threshold: float = 2.0


class DocumentSearchSettings(BaseModel):
    """
    Document-level search settings.

    chunk_retrieval_multiplier: How many chunks to fetch per requested document.
    """

    chunk_retrieval_multiplier: int = 5


class RerankerSettings(BaseModel):
    """
    Cross-encoder reranker settings.

    CrossEncoder logits typically range from -12 to +12.
    Positive = relevant, negative = irrelevant.
    0.0 is the decision boundary (50% relevance probability).

    Note: Score thresholds are configured per-search-type in
    ClientSearchSettings and ChunkSearchSettings.
    """

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RRFSettings(BaseModel):
    """
    Reciprocal Rank Fusion settings.

    k: Smoothing constant that reduces impact of high rankings.
    Default 60 is the standard value from the original RRF paper.
    """

    k: int = 60


class EmbeddingSettings(BaseModel):
    """Embedding model settings."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


class ChunkingSettings(BaseModel):
    """
    Text chunking settings for document processing.

    Values are in tokens (using the embedding model's tokenizer).
    """

    size: int = 256
    overlap: int = 25


class SummarizationSettings(BaseModel):
    """
    Document summarization settings.

    max_words: Target word count for summaries.
    max_tokens: Maximum tokens for LLM response.
    """

    enabled: bool = True
    provider: str = "claude"  # Options: "claude", "gemini"
    max_words: int = 100
    max_tokens: int = 200


class LLMSettings(BaseModel):
    """LLM provider settings and API keys."""

    claude_model: str = "claude-sonnet-4-20250514"
    gemini_model: str = "models/gemini-flash-latest"
    anthropic_api_key: str | None = None
    google_api_key: str | None = None


class S3Settings(BaseModel):
    """S3 storage settings."""

    bucket_name: str = "nevis-documents"
    endpoint_url: str | None = None  # For LocalStack: http://localhost:4566
    document_key_pattern: str = "clients/{client_id}/documents/{document_id}.txt"


class AWSSettings(BaseModel):
    """AWS credentials and region settings."""

    region: str = "eu-west-1"
    access_key_id: str | None = None
    secret_access_key: str | None = None


# =============================================================================
# Main Settings Class
# =============================================================================


class Settings(BaseSettings):
    """
    Application settings with nested configuration groups.

    Environment variables use double underscore as delimiter for nested values.
    Example: SEARCH__DEFAULT_TOP_K=5, EMBEDDING__MODEL_NAME=sentence-transformers/all-mpnet-base-v2
    """

    # Application metadata
    app_name: str = "Nevis API"
    app_version: str = "1.0.0"

    # Database
    database_url: str = "postgresql+asyncpg://localhost/nevis"

    # Nested settings groups
    search: SearchSettings = SearchSettings()
    client_search: ClientSearchSettings = ClientSearchSettings()
    chunk_search: ChunkSearchSettings = ChunkSearchSettings()
    document_search: DocumentSearchSettings = DocumentSearchSettings()
    reranker: RerankerSettings = RerankerSettings()
    rrf: RRFSettings = RRFSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    summarization: SummarizationSettings = SummarizationSettings()
    llm: LLMSettings = LLMSettings()
    s3: S3Settings = S3Settings()
    aws: AWSSettings = AWSSettings()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def summarization_available(self) -> bool:
        """Check if summarization is enabled and has required API key."""
        if not self.summarization.enabled:
            return False
        if self.summarization.provider == "claude" and self.llm.anthropic_api_key:
            return True
        if self.summarization.provider == "gemini" and self.llm.google_api_key:
            return True
        return False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
