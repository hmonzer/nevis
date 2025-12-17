"""Domain models used in business logic."""
import uuid
from datetime import datetime
from enum import StrEnum
from typing import Generic, Literal, TypeVar, Union
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, field_validator


class Client(BaseModel):
    """Domain model for Client used in business logic."""
    id: UUID = Field(default_factory=uuid.uuid4, description="Unique client ID")
    first_name: str = Field(..., min_length=1, description="First name cannot be blank")
    last_name: str = Field(..., min_length=1, description="Last name cannot be blank")
    email: EmailStr = Field(..., description="Email address is required")
    description: str | None = None
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")

    model_config = {"from_attributes": True}


class DocumentStatus(StrEnum):
    """Document processing status."""
    PENDING = "PENDING"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"


class Document(BaseModel):
    """Domain model for Document used in business logic."""
    id: UUID
    client_id: UUID = Field(..., description="ID of the client who owns this document")
    title: str = Field(..., min_length=1, description="Document title")
    s3_key: str = Field(..., description="S3 path to the stored document")
    status: DocumentStatus = Field(default=DocumentStatus.PENDING, description="Processing status")
    summary: str | None = Field(default=None, description="AI-generated summary of the document")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")

    model_config = {"from_attributes": True}

    def processed(self) -> None:
        self.status = DocumentStatus.PROCESSED

    def failed(self) -> None:
        self.status = DocumentStatus.FAILED

    def summarized(self, summary: str) -> None:
        """Set the document summary."""
        self.summary = summary if summary else None


class DocumentChunk(BaseModel):
    """Domain model for DocumentChunk used in business logic."""

    id: UUID = Field(default_factory=uuid.uuid4, description="Unique chunk ID")
    document_id: UUID = Field(..., description="ID of the parent document")
    chunk_index: int = Field(..., ge=0, description="Index of this chunk in the document")
    chunk_content: str = Field(..., min_length=1, description="Text content of this chunk")
    embedding: list[float] | None = Field(default=None, description="Vector embedding (null in Phase 1)")

    model_config = {"from_attributes": True}


# =============================================================================
# Unified Scoring Types
# =============================================================================

class ScoreSource(StrEnum):
    """
    Origin of a relevance score for interpretability and debugging.

    Each source has a different score range and semantic meaning:
    - VECTOR_SIMILARITY: Cosine similarity [-1.0, 1.0]
    - KEYWORD_RANK: PostgreSQL ts_rank [0, unbounded]
    - TRIGRAM_SIMILARITY: pg_trgm similarity [0.0, 1.0]
    - RRF_FUSION: Reciprocal Rank Fusion score [small positive floats]
    - CROSS_ENCODER: CrossEncoder logits [~-12, +12], 0 = 50% relevance
    """
    VECTOR_SIMILARITY = "vector_similarity"
    KEYWORD_RANK = "keyword_rank"
    TRIGRAM_SIMILARITY = "trigram_similarity"
    RRF_FUSION = "rrf_fusion"
    CROSS_ENCODER = "cross_encoder"

    def of(self, value: float) -> "Score":
        return Score(value=value, source=self)


class Score(BaseModel):
    """
    A relevance score with its source/origin.

    Encapsulates both the numeric value and where it came from,
    enabling meaningful interpretation and debugging. Immutable
    for safe use in score history tracking.
    """
    value: float = Field(..., description="The numeric score value")
    source: ScoreSource = Field(..., description="Origin of this score")

    model_config = {"frozen": True}

    def __repr__(self) -> str:
        return f"Score({self.value:.4f}, {self.source.value})"


T = TypeVar("T")


class ScoredResult(BaseModel, Generic[T]):
    """
    Universal wrapper for any entity with a relevance score and history.

    Same type throughout the entire retrieval pipeline. Tracks how the
    score evolves through different stages (retrieval → fusion → reranking).

    Example history for a chunk going through the pipeline:
    1. Initial vector search: Score(0.85, VECTOR_SIMILARITY), history=[]
    2. After RRF fusion: Score(0.032, RRF_FUSION), history=[Score(0.85, VECTOR_SIMILARITY)]
    3. After reranking: Score(4.2, CROSS_ENCODER), history=[..., Score(0.032, RRF_FUSION)]

    Attributes:
        item: The entity being scored (DocumentChunk, Client, Document, etc.)
        score: Current relevance score with source metadata
        score_history: Previous scores in chronological order (oldest first)
    """
    item: T
    score: Score = Field(..., description="Current score with source")
    score_history: list[Score] = Field(
        default_factory=list,
        description="Previous scores in chronological order (oldest first)"
    )

    model_config = {"from_attributes": True, "arbitrary_types_allowed": True}

    def assign_score(self, new_score: Score) -> "ScoredResult[T]":
        """
        Assign a new score, preserving the current score in history.

        This enables middleware pattern while tracking score evolution:
        - Current score moves to end of score_history
        - New score becomes the current score
        - Returns new immutable ScoredResult (does not mutate self)

        Args:
            new_score: The new Score to assign

        Returns:
            New ScoredResult with updated score and preserved history
        """
        return ScoredResult(
            item=self.item,
            score=new_score,
            score_history=[*self.score_history, self.score]
        )

    @property
    def value(self) -> float:
        """Convenience accessor for current score value."""
        return self.score.value

    @property
    def source(self) -> ScoreSource:
        """Convenience accessor for current score source."""
        return self.score.source

    @staticmethod
    def filter_by_threshold(
        results: list["ScoredResult[T]"],
        threshold: float
    ) -> list["ScoredResult[T]"]:
        """
        Filter results below a score threshold.

        Args:
            results: List of ScoredResult to filter
            threshold: Minimum score value (inclusive)

        Returns:
            Filtered list with only results where score.value >= threshold
        """
        return [r for r in results if r.value >= threshold]


# =============================================================================
# Search Request Model
# =============================================================================

class SearchRequest(BaseModel):
    """
    Request model for search operations across all search services.

    Provides centralized validation for common search parameters.
    Thresholds are configured per-component in application settings.
    """
    query: str = Field(..., min_length=1, description="Search query string")
    top_k: int = Field(default=10, gt=0, le=100, description="Maximum number of results to return")

    @field_validator("query")
    @classmethod
    def validate_query_not_blank(cls, v: str) -> str:
        """Ensure query is not just whitespace."""
        if not v.strip():
            raise ValueError("Search query cannot be empty or whitespace only")
        return v.strip()


class SearchResult(BaseModel):
    """
    Unified search result that can contain either a Client or Document.

    Used by the unified SearchService to return heterogeneous search results
    sorted by score descending.
    """
    type: Literal["CLIENT", "DOCUMENT"] = Field(..., description="Type of entity in the result")
    entity: Union[Client, Document] = Field(..., description="The actual entity (Client or Document)")
    score: float = Field(..., description="Relevance score from the search")

    model_config = {"from_attributes": True}
