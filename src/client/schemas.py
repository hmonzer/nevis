"""API schemas for client requests and responses."""
from datetime import datetime
from uuid import UUID
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
