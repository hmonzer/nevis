import asyncio

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport, HTTPStatusError
from pydantic.v1 import EmailStr
from testcontainers.postgres import PostgresContainer

from src.app.main import create_app
from src.client import NevisClient, CreateClientRequest
from src.shared.database.database import Database, Base
from src.shared.database.database_settings import DatabaseSettings


@pytest.fixture(scope="module")
def postgres_container():
    """Start a PostgreSQL container for testing."""
    with PostgresContainer("postgres:16-alpine") as postgres:
        yield postgres


@pytest_asyncio.fixture(scope="function")
async def test_app(postgres_container):
    """Create test application with test database."""
    connection_url = postgres_container.get_connection_url()
    async_url = connection_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://")

    # Create database tables
    db_settings = DatabaseSettings(db_url=async_url)
    db = Database(db_settings)
    await wait_till_db_ready(db)

    async with db._engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    await db._engine.dispose()

    # Create app
    app = create_app()

    # Override the get_database dependency to use test database
    from src.app.api.dependencies import get_database

    async def override_get_database():
        test_db = Database(db_settings)
        try:
            yield test_db
        finally:
            await test_db._engine.dispose()

    app.dependency_overrides[get_database] = override_get_database

    yield app

    # Clear overrides
    app.dependency_overrides.clear()


async def wait_till_db_ready(db):
    """Wait for database to be ready."""
    for attempt in range(10):
        try:
            async with db._engine.begin():
                return
        except Exception:
            await asyncio.sleep(0.1)
    raise Exception("Database not ready after 10 attempts")


@pytest_asyncio.fixture
async def nevis_client(test_app):
    """Create a Nevis client for testing."""
    transport = ASGITransport(app=test_app)
    http_client = AsyncClient(transport=transport, base_url="http://test")
    client = NevisClient(base_url="http://test", client=http_client)

    async with client:
        yield client


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
    assert exc_info.value.response.status_code == 400
    assert "already exists" in exc_info.value.response.json()["detail"]


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
    """Test that creating a client with blank first name fails validation."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc_info:
        CreateClientRequest(
            first_name="",
            last_name="Doe",
            email=EmailStr("test@example.com"),
            description="Test"
        )

    errors = exc_info.value.errors()
    assert any(error["loc"] == ("first_name",) for error in errors)


@pytest.mark.asyncio
async def test_create_client_blank_last_name():
    """Test that creating a client with blank last name fails validation."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc_info:
        CreateClientRequest(
            first_name="John",
            last_name="",
            email=EmailStr("test@example.com"),
            description="Test"
        )

    errors = exc_info.value.errors()
    assert any(error["loc"] == ("last_name",) for error in errors)


@pytest.mark.asyncio
async def test_create_client_whitespace_only_names():
    """Test that creating a client with whitespace-only names fails validation."""
    from pydantic import ValidationError

    # Whitespace-only first name
    with pytest.raises(ValidationError) as exc_info:
        CreateClientRequest(
            first_name="   ",
            last_name="Doe",
            email=EmailStr("test@example.com"),
            description="Test"
        )

    errors = exc_info.value.errors()
    assert any(error["loc"] == ("first_name",) for error in errors)

    # Whitespace-only last name
    with pytest.raises(ValidationError) as exc_info:
        CreateClientRequest(
            first_name="John",
            last_name="   ",
            email=EmailStr("test@example.com"),
            description="Test"
        )

    errors = exc_info.value.errors()
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
