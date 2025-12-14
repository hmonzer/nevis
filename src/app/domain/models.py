"""Domain models used in business logic."""
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, EmailStr, Field, field_validator


class Client(BaseModel):
    """Domain model for Client used in business logic."""
    id: UUID
    first_name: str = Field(..., min_length=1, description="First name cannot be blank")
    last_name: str = Field(..., min_length=1, description="Last name cannot be blank")
    email: EmailStr = Field(..., description="Email address is required")
    description: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}
