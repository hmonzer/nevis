"""API schemas for client and document requests and responses."""
from datetime import datetime
from typing import Union
from uuid import UUID
from enum import Enum
from pydantic import BaseModel, EmailStr, Field, field_validator


class CreateClientRequest(BaseModel):
    """Request schema for creating a new client."""
    first_name: str = Field(..., min_length=1, description="First name cannot be blank")
    last_name: str = Field(..., min_length=1, description="Last name cannot be blank")
    email: EmailStr = Field(..., description="Email address is required")
    description: str | None = None

    @field_validator("first_name", "last_name")
    @classmethod
    def validate_not_blank(cls, v: str) -> str:
        """Ensure name fields are not just whitespace."""
        if not v or not v.strip():
            raise ValueError("Field cannot be blank or only whitespace")
        return v.strip()


class ClientResponse(BaseModel):
    """Response schema for client data returned by the API."""
    id: UUID
    first_name: str = Field(..., min_length=1)
    last_name: str = Field(..., min_length=1)
    email: EmailStr = Field(..., description="Email address")
    description: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}

    @field_validator("first_name", "last_name")
    @classmethod
    def validate_not_blank(cls, v: str) -> str:
        """Ensure name fields are not just whitespace."""
        if not v or not v.strip():
            raise ValueError("Field cannot be blank or only whitespace")
        return v.strip()


class DocumentStatusEnum(str, Enum):
    """Document processing status enum for API."""
    PENDING = "PENDING"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"


class CreateDocumentRequest(BaseModel):
    """Request schema for creating a new document."""
    title: str = Field(..., min_length=1, description="Document title")
    content: str = Field(..., min_length=1, description="Text content of the document")

    @field_validator("title", "content")
    @classmethod
    def validate_not_blank(cls, v: str) -> str:
        """Ensure fields are not just whitespace."""
        if not v or not v.strip():
            raise ValueError("Field cannot be blank or only whitespace")
        return v.strip()


class DocumentResponse(BaseModel):
    """Response schema for document data returned by the API."""
    id: UUID
    client_id: UUID
    title: str
    s3_key: str
    status: DocumentStatusEnum
    summary: str | None = Field(default=None, description="AI-generated summary of the document")
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentDownloadResponse(BaseModel):
    """Response schema for document download URL."""
    id: UUID
    title: str
    download_url: str = Field(..., description="Pre-signed S3 URL for downloading the document")
    expires_in: int = Field(..., description="URL expiration time in seconds")


class SearchResultTypeEnum(str, Enum):
    """Type of entity in a search result."""
    CLIENT = "CLIENT"
    DOCUMENT = "DOCUMENT"


class SearchResultResponse(BaseModel):
    """
    Response schema for a unified search result.

    Can contain either a Client or Document entity with its relevance score.
    Results are sorted by score descending.
    """
    type: SearchResultTypeEnum = Field(..., description="Type of entity in the result")
    entity: Union[ClientResponse, DocumentResponse] = Field(
        ..., description="The matched entity (Client or Document)"
    )
    score: float = Field(..., description="Relevance score (higher is more relevant)")

    model_config = {"from_attributes": True}
