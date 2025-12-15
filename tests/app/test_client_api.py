from typing import cast

import pytest
from httpx import HTTPStatusError
from pydantic import ValidationError
from pydantic.v1 import EmailStr

from src.client import CreateClientRequest


@pytest.mark.asyncio
async def test_create_client(nevis_client):
    """Test creating a client via API."""
    request = CreateClientRequest(
        first_name="John",
        last_name="Doe",
        email=EmailStr("john.doe@example.com"),
        description="Test client"
    )

    response = await nevis_client.create_client(request)

    assert response.first_name == "John"
    assert response.last_name == "Doe"
    assert response.email == "john.doe@example.com"
    assert response.description == "Test client"
    assert response.id is not None
    assert response.created_at is not None


@pytest.mark.asyncio
async def test_create_duplicate_email(nevis_client):
    """Test creating a client with duplicate email fails."""
    # Create first client
    first_request = CreateClientRequest(
        first_name="Jane",
        last_name="Smith",
        email=EmailStr("jane@example.com"),
        description="First client"
    )
    await nevis_client.create_client(first_request)

    # Try to create second client with same email
    duplicate_request = CreateClientRequest(
        first_name="Jane",
        last_name="Doe",
        email=EmailStr("jane@example.com"),
        description="Second client"
    )

    with pytest.raises(HTTPStatusError) as exc_info:
        await nevis_client.create_client(duplicate_request)

    error = cast(HTTPStatusError, exc_info.value)
    assert error.response.status_code == 409
    assert "already exists" in error.response.json()["detail"]


@pytest.mark.asyncio
async def test_get_client(nevis_client):
    """Test getting a client by ID."""
    # Create a client
    request = CreateClientRequest(
        first_name="Bob",
        last_name="Johnson",
        email=EmailStr("bob@example.com"),
        description="Test client"
    )
    created_response = await nevis_client.create_client(request)

    # Get the client
    retrieved_response = await nevis_client.get_client(created_response.id)

    assert retrieved_response.id == created_response.id
    assert retrieved_response.first_name == "Bob"
    assert retrieved_response.last_name == "Johnson"
    assert retrieved_response.email == "bob@example.com"


@pytest.mark.asyncio
async def test_create_client_blank_first_name():
    """Test that creating a client with a blank first name fails validation."""
    with pytest.raises(ValidationError) as exc_info:
        CreateClientRequest(
            first_name="",
            last_name="Doe",
            email=EmailStr("test@example.com"),
            description="Test"
        )

    error = cast(ValidationError, exc_info.value)
    errors = error.errors()
    assert any(error["loc"] == ("first_name",) for error in errors)


@pytest.mark.asyncio
async def test_create_client_blank_last_name():
    """Test that creating a client with a blank last name fails validation."""
    with pytest.raises(ValidationError) as exc_info:
        CreateClientRequest(
            first_name="John",
            last_name="",
            email=EmailStr("test@example.com"),
            description="Test"
        )

    error = cast(ValidationError, exc_info.value)
    errors = error.errors()
    assert any(error["loc"] == ("last_name",) for error in errors)


@pytest.mark.asyncio
async def test_create_client_whitespace_only_names():
    """Test that creating a client with whitespace-only names fails validation."""
    # Whitespace-only first name
    with pytest.raises(ValidationError) as exc_info:
        CreateClientRequest(
            first_name="   ",
            last_name="Doe",
            email=EmailStr("test@example.com"),
            description="Test"
        )

    error = cast(ValidationError, exc_info.value)
    errors = error.errors()
    assert any(error["loc"] == ("first_name",) for error in errors)

    # Whitespace-only last name
    with pytest.raises(ValidationError) as exc_info:
        CreateClientRequest(
            first_name="John",
            last_name="   ",
            email=EmailStr("test@example.com"),
            description="Test"
        )

    error = cast(ValidationError, exc_info.value)
    errors = error.errors()
    assert any(error["loc"] == ("last_name",) for error in errors)


@pytest.mark.asyncio
async def test_create_client_trims_whitespace():
    """Test that names are trimmed of leading/trailing whitespace."""
    request = CreateClientRequest(
        first_name="  John  ",
        last_name="  Doe  ",
        email=EmailStr("test@example.com"),
        description="Test"
    )

    assert request.first_name == "John"
    assert request.last_name == "Doe"
