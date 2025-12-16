"""Domain models used in business logic."""
import uuid
from datetime import datetime
from enum import StrEnum
from typing import Literal, Union
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


class ChunkSearchResult(BaseModel):
    """
    Domain model representing a document chunk search result with relevance score.

    This model pairs a DocumentChunk with its similarity/relevance score.
    The score can come from different sources:
    - Cosine similarity: typically in range [-1.0, 1.0]
    - CrossEncoder reranking: unbounded logits (any real number)

    Higher scores always indicate higher relevance, regardless of source.
    """
    chunk: DocumentChunk = Field(..., description="The document chunk that matched the search")
    score: float = Field(..., description="Relevance score (higher is more relevant). Range depends on scoring method.")

    model_config = {"from_attributes": True}


class ClientSearchResult(BaseModel):
    """
    Domain model representing a client search result with relevance score.

    This model pairs a Client with their fuzzy match similarity score from
    PostgreSQL's pg_trgm extension, allowing search results to be ranked by relevance.
    """
    client: Client = Field(..., description="The client that matched the search")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score (0.0 to 1.0, higher is more relevant)")

    model_config = {"from_attributes": True}


class DocumentSearchResult(BaseModel):
    """
    Domain model representing a document search result with relevance score.

    This model aggregates chunk-level search results to the document level,
    using the highest score among all matching chunks as the document's relevance score.
    """
    document: Document = Field(..., description="The document that matched the search")
    score: float = Field(..., description="Relevance score (higher is more relevant). Highest score from matching chunks.")

    model_config = {"from_attributes": True}


class SearchRequest(BaseModel):
    """
    Request model for search operations across all search services.

    Provides centralized validation for common search parameters.
    """
    query: str = Field(..., min_length=1, description="Search query string")
    top_k: int = Field(default=10, gt=0, le=100, description="Maximum number of results to return")
    threshold: float = Field(default=0.5, ge=-1.0, le=1.0, description="Minimum similarity threshold for results")

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
    with consistent ranking and scoring.
    """
    type: Literal["CLIENT", "DOCUMENT"] = Field(..., description="Type of entity in the result")
    entity: Union[Client, Document] = Field(..., description="The actual entity (Client or Document)")
    score: float = Field(..., description="Relevance score from the search")
    rank: int = Field(..., ge=1, description="Rank position in the result set (1-based)")

    model_config = {"from_attributes": True}
