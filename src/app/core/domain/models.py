"""Domain models used in business logic."""
import uuid
from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


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
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")

    model_config = {"from_attributes": True}

    def processed(self) -> None:
        self.status = DocumentStatus.PROCESSED

    def failed(self) -> None:
        self.status = DocumentStatus.FAILED


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
