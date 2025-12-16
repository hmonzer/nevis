"""Shared test fixtures and utilities for all tests."""
import asyncio
import os

import pytest
import pytest_asyncio
from dependency_injector import providers
from httpx import ASGITransport, AsyncClient
from testcontainers.postgres import PostgresContainer
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs
from sqlalchemy import text

from src.app.containers import Container
from src.client import NevisClient
from src.shared.database.database import Database, Base, DatabaseSettings
from src.shared.database.unit_of_work import UnitOfWork
from src.shared.blob_storage.s3_blober import S3BlobStorage, S3BlobStorageSettings


@pytest.fixture(scope="module")
def postgres_container():
    """Start a PostgreSQL container with pgvector for testing. Module-scoped for reuse."""
    with PostgresContainer("pgvector/pgvector:pg16") as postgres:
        yield postgres


@pytest.fixture(scope="module")
def async_db_url(postgres_container):
    """
    Get the async database URL from the postgres container.
    Module-scoped so it can be reused across tests.
    """
    connection_url = postgres_container.get_connection_url()
    async_url = connection_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://")
    return async_url


@pytest.fixture(scope="module")
def localstack_container():
    """
    Start a LocalStack container for testing S3.
    Module-scoped for reuse across tests.
    """
    container = (
        DockerContainer("localstack/localstack:latest")
        .with_exposed_ports(4566)
        .with_env("SERVICES", "s3")
        .with_env("DEFAULT_REGION", "us-east-1")
        .with_env("AWS_ACCESS_KEY_ID", "test")
        .with_env("AWS_SECRET_ACCESS_KEY", "test")
    )

    with container:
        # Wait for LocalStack to be ready
        wait_for_logs(container, "Ready.", timeout=30)
        yield container


@pytest.fixture(scope="module")
def s3_endpoint_url(localstack_container):
    """
    Get the S3 endpoint URL from LocalStack container.
    Module-scoped for reuse across tests.
    """
    host = localstack_container.get_container_host_ip()
    port = localstack_container.get_exposed_port(4566)
    return f"http://{host}:{port}"


@pytest.fixture(scope="module")
def test_settings_override(async_db_url, s3_endpoint_url):
    """
    Centralized settings override for all test configurations.
    Module-scoped to set up environment once per test module.

    This fixture manages all environment variable overrides needed for testing,
    providing a single place to configure test settings.
    """
    os.environ["DATABASE_URL"] = async_db_url
    os.environ["S3_ENDPOINT_URL"] = s3_endpoint_url
    os.environ["S3_BUCKET_NAME"] = "test-documents"
    os.environ["AWS_ACCESS_KEY_ID"] = "test"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "test"

    # Clear settings cache to force reload with new env vars
    from src.app.config import get_settings
    get_settings.cache_clear()

    yield

    get_settings.cache_clear()


async def wait_till_db_ready(db: Database, max_attempts: int = 20):
    """
    Wait for database to be ready.

    Args:
        db: Database instance to test
        max_attempts: Maximum number of connection attempts

    Raises:
        Exception: If database is not ready after max_attempts
    """
    for attempt in range(max_attempts):
        try:
            async with db._engine.begin():
                return
        except Exception:
            await asyncio.sleep(0.2)
    raise Exception(f"Database not ready after {max_attempts} attempts")


@pytest_asyncio.fixture(scope="function")
async def db(async_db_url):
    """
    Create database instance with test database.
    Function-scoped for test isolation.
    """
    db_settings = DatabaseSettings(db_url=async_db_url)
    db = Database(db_settings)
    await wait_till_db_ready(db)
    yield db
    await db._engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def clean_database(db):
    """
    Clean the database before each test.
    Drops and recreates all tables.
    """
    async with db._engine.begin() as conn:
        # Enable pgvector extension
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # Enable pg_trgm extension for fuzzy search
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))

        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield db


@pytest.fixture(scope="function")
def test_container(test_settings_override, clean_database):
    """
    Create a test container with database override for proper test isolation.
    Function-scoped to ensure each test gets a fresh container.

    Overrides the container's database singleton with the test database.
    """
    container = Container()

    # Override the database singleton with the test database instance
    container.database.override(providers.Object(clean_database))

    container.wire(modules=[
        "src.app.api.v1.clients",
        "src.app.api.v1.documents",
        "src.app.api.v1.search",
    ])
    yield container
    container.database.reset_override()
    container.unwire()


@pytest_asyncio.fixture(scope="function")
async def test_app(test_container):
    """
    Create test application with container.
    Function-scoped for test isolation.
    """
    from fastapi import FastAPI
    from contextlib import asynccontextmanager
    from src.app.api.v1 import clients, documents, search

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Database tables are already created by clean_database fixture
        yield

    config = test_container.config()
    app = FastAPI(
        title=config.app_name,
        version=config.app_version,
        lifespan=lifespan
    )
    app.state.container = test_container

    app.include_router(clients.router, prefix="/api/v1")
    app.include_router(documents.router, prefix="/api/v1")
    app.include_router(search.router, prefix="/api/v1")

    @app.get("/")
    async def root():
        return {"message": "Welcome to Nevis API"}

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    yield app


@pytest_asyncio.fixture
async def nevis_client(test_app):
    """
    Create a Nevis client for testing.
    test_app already depends on clean_database for test isolation.
    """
    transport = ASGITransport(app=test_app)
    http_client = AsyncClient(transport=transport, base_url="http://test")
    client = NevisClient(base_url="http://test", client=http_client)

    async with client:
        yield client


@pytest_asyncio.fixture(scope="function")
async def unit_of_work(clean_database, test_container):
    """
    Fixture for a UnitOfWork instance with a clean database.
    Uses the container's entity_mapper singleton.
    """
    entity_mapper = test_container.entity_mapper()
    yield UnitOfWork(clean_database, entity_mapper)


# =========================================================================
# Service fixtures from container (for direct service testing)
# =========================================================================

@pytest.fixture
def s3_storage(s3_endpoint_url):
    """
    Get S3 blob storage instance configured for LocalStack.
    """
    settings = S3BlobStorageSettings(
        bucket_name="test-documents",
        endpoint_url=s3_endpoint_url,
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    return S3BlobStorage(settings)


# =========================================================================
# Common repository fixtures (available to all test directories)
# =========================================================================

@pytest_asyncio.fixture
def client_repository(test_container):
    """Get client repository from container."""
    return test_container.client_repository()


@pytest_asyncio.fixture
def document_repository(test_container):
    """Get document repository from container."""
    return test_container.document_repository()


@pytest_asyncio.fixture
def document_chunk_repository(test_container):
    """Get document chunk repository from container."""
    return test_container.document_chunk_repository()


@pytest_asyncio.fixture
def chunks_search_repository(test_container):
    """Get chunks search repository from container."""
    return test_container.chunks_search_repository()


@pytest_asyncio.fixture
def client_search_repository(test_container):
    """Get client search repository from container."""
    return test_container.client_search_repository()


# =========================================================================
# Common service fixtures (available to all test directories)
# =========================================================================

@pytest_asyncio.fixture
def unit_of_work_fixture(test_container):
    """Get unit of work from container."""
    return test_container.unit_of_work()


@pytest_asyncio.fixture
def document_service(test_container):
    """Get document service from container."""
    return test_container.document_service()


@pytest_asyncio.fixture
def client_service(test_container):
    """Get client service from container."""
    return test_container.client_service()


@pytest_asyncio.fixture
def embedding_service(test_container):
    """Get embedding service from container."""
    return test_container.embedding_service()


@pytest_asyncio.fixture
def chunk_search_service(test_container):
    """Get document chunk search service from container."""
    return test_container.document_chunk_search_service()


@pytest_asyncio.fixture
def document_search_service(test_container):
    """Get document search service from container."""
    return test_container.document_search_service()


@pytest_asyncio.fixture
def client_search_service(test_container):
    """Get client search service from container."""
    return test_container.client_search_service()


@pytest_asyncio.fixture
def search_service(test_container):
    """Get unified search service WITH reranking from container."""
    return test_container.search_service()


@pytest_asyncio.fixture
def search_service_no_rerank(test_container):
    """Get unified search service WITHOUT reranking from container."""
    return test_container.search_service_no_rerank()
