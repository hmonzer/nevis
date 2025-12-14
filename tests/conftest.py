"""Shared test fixtures and utilities for all tests."""
import asyncio
import os

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from testcontainers.postgres import PostgresContainer
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs
from sqlalchemy import text

from src.app.main import create_app
from src.client import NevisClient
from src.shared.database.database import Database, Base, DatabaseSettings
from src.shared.database.unit_of_work import UnitOfWork
from src.shared.database.entity_mapper import EntityMapper
from src.app.infrastructure.mappers.client_mapper import ClientMapper
from src.app.infrastructure.mappers.document_mapper import DocumentMapper
from src.app.infrastructure.mappers.document_chunk_mapper import DocumentChunkMapper
from src.app.core.domain.models import Client, Document, DocumentChunk
from src.shared.blob_storage.s3_blober import S3BlobStorage, S3BlobStorageSettings

@pytest.fixture(scope="session")
def get_entity_mapper_fixture():
    return EntityMapper(
        entity_mappings={
            Client: ClientMapper().to_entity,
            Document: DocumentMapper().to_entity,
            DocumentChunk: DocumentChunkMapper().to_entity,
        }
    )


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
def s3_storage(s3_endpoint_url):
    """
    Get S3 blob storage instance configured for LocalStack.
    Module-scoped for reuse across tests.
    """
    settings = S3BlobStorageSettings(
        bucket_name="test-documents",
        endpoint_url=s3_endpoint_url,
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    return S3BlobStorage(settings)


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
            await asyncio.sleep(0.2)  # Increased sleep time
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

        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield db


@pytest_asyncio.fixture(scope="module")
async def test_app(test_settings_override):
    """
    Create test application with test database.
    Module-scoped so the app is created once per test module for better performance.
    Uses centralized test_settings_override fixture for configuration.
    """
    app = create_app()
    yield app


@pytest_asyncio.fixture
async def nevis_client(test_app, clean_database):
    """
    Create a Nevis client for testing.
    Depends on clean_database to ensure test isolation between runs.
    """
    transport = ASGITransport(app=test_app)
    http_client = AsyncClient(transport=transport, base_url="http://test")
    client = NevisClient(base_url="http://test", client=http_client)

    async with client:
        yield client


@pytest_asyncio.fixture(scope="function")
async def unit_of_work(clean_database, get_entity_mapper_fixture):
    """
    Fixture for a UnitOfWork instance with a clean database.
    """
    yield UnitOfWork(clean_database, get_entity_mapper_fixture)
