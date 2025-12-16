"""Shared test fixtures and utilities for all tests."""
import os

# Disable tokenizers parallelism to avoid fork warnings from HuggingFace
# This must be set before any tokenizers are imported
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import asyncio

import pytest
import pytest_asyncio
from dependency_injector import providers
from httpx import ASGITransport, AsyncClient
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs
from testcontainers.postgres import PostgresContainer

from src.app.containers import Container
from src.app.main import create_app, default_lifespan
from src.client import NevisClient
from src.shared.blob_storage.s3_blober import S3BlobStorage, S3BlobStorageSettings
from src.shared.database.database import Database, Base, DatabaseSettings
from src.shared.database.unit_of_work import UnitOfWork


# =============================================================================
# Session-scoped infrastructure fixtures (expensive to create, shared across all tests)
# =============================================================================

@pytest.fixture(scope="session")
def postgres_container():
    """Start a PostgreSQL container with pgvector for testing. Session-scoped for reuse."""
    with PostgresContainer("pgvector/pgvector:pg16") as postgres:
        yield postgres


@pytest.fixture(scope="session")
def async_db_url(postgres_container):
    """Get the async database URL from the postgres container."""
    connection_url = postgres_container.get_connection_url()
    async_url = connection_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://")
    return async_url


@pytest.fixture(scope="session")
def localstack_container():
    """Start a LocalStack container for testing S3. Session-scoped for reuse."""
    container = (
        DockerContainer("localstack/localstack:latest")
        .with_exposed_ports(4566)
        .with_env("SERVICES", "s3")
        .with_env("DEFAULT_REGION", "eu-west-1")
        .with_env("AWS_ACCESS_KEY_ID", "test")
        .with_env("AWS_SECRET_ACCESS_KEY", "test")
    )

    with container:
        wait_for_logs(container, "Ready.", timeout=30)
        yield container


@pytest.fixture(scope="session")
def s3_endpoint_url(localstack_container):
    """Get the S3 endpoint URL from LocalStack container."""
    host = localstack_container.get_container_host_ip()
    port = localstack_container.get_exposed_port(4566)
    return f"http://{host}:{port}"


@pytest.fixture(scope="session")
def test_settings_override(async_db_url, s3_endpoint_url):
    """
    Centralized settings override for all test configurations.
    Session-scoped to set up environment once for all tests.

    Note: With nested config using env_nested_delimiter="__",
    nested settings use double underscore (e.g., S3__BUCKET_NAME).
    """
    os.environ["DATABASE_URL"] = async_db_url
    os.environ["S3__ENDPOINT_URL"] = s3_endpoint_url
    os.environ["S3__BUCKET_NAME"] = "test-documents"
    os.environ["AWS__ACCESS_KEY_ID"] = "test"
    os.environ["AWS__SECRET_ACCESS_KEY"] = "test"

    # Clear settings cache to force reload with new env vars
    from src.app.config import get_settings
    get_settings.cache_clear()

    yield

    get_settings.cache_clear()


# =============================================================================
# Session-scoped database fixture
# =============================================================================

async def wait_till_db_ready(db: Database, max_attempts: int = 20):
    """Wait for database to be ready."""
    for attempt in range(max_attempts):
        try:
            async with db._engine.begin():
                return
        except Exception:
            await asyncio.sleep(0.2)
    raise Exception(f"Database not ready after {max_attempts} attempts")


@pytest_asyncio.fixture(scope="session")
async def session_db(async_db_url):
    """
    Session-scoped database instance.
    Only waits for database readiness - extensions and tables are created
    by the app lifespan in test_app fixture.
    """
    db_settings = DatabaseSettings(db_url=async_db_url)
    db = Database(db_settings)
    await wait_till_db_ready(db)

    yield db

    await db._engine.dispose()


# =============================================================================
# Session-scoped container and app fixtures
# =============================================================================

@pytest.fixture(scope="session")
def test_container(test_settings_override, session_db):
    """
    Session-scoped test container with database override.
    Reuses expensive ML model singletons across all tests.
    """
    container = Container()

    # Override the database singleton with the test database instance
    container.database.override(providers.Object(session_db))

    yield container

    container.database.reset_override()
    container.unwire()


@pytest_asyncio.fixture(scope="session")
async def test_app(test_container):
    """
    Session-scoped test application using create_app() from src/app/main.py.
    Explicitly invokes lifespan to initialize database extensions, tables, and ML models.
    """
    app = create_app(container=test_container)

    # Explicitly invoke lifespan to create extensions, tables, and load ML models
    async with default_lifespan(app):
        yield app


# =============================================================================
# Function-scoped fixtures for test isolation
# =============================================================================

@pytest_asyncio.fixture(scope="function")
async def clean_database(session_db, test_app):
    """
    Clean the database before each test by dropping and recreating all tables.
    Depends on test_app to ensure extensions are created first via app lifespan.
    """
    # test_app dependency ensures lifespan has run (extensions + initial tables created)
    _ = test_app

    async with session_db._engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    yield session_db


@pytest_asyncio.fixture
async def nevis_client(test_app, clean_database):
    """
    Create a Nevis client for testing.
    Depends on clean_database for test isolation.
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


# =============================================================================
# Aliases for backwards compatibility
# =============================================================================

@pytest_asyncio.fixture(scope="function")
async def db(clean_database):
    """Alias for clean_database for backwards compatibility."""
    yield clean_database


# =============================================================================
# Service fixtures from container (for direct service testing)
# =============================================================================

@pytest.fixture
def s3_storage(s3_endpoint_url):
    """Get S3 blob storage instance configured for LocalStack."""
    settings = S3BlobStorageSettings(
        bucket_name="test-documents",
        endpoint_url=s3_endpoint_url,
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    return S3BlobStorage(settings)


# =============================================================================
# Common repository fixtures (available to all test directories)
# =============================================================================

@pytest.fixture
def client_repository(test_container):
    """Get client repository from container."""
    return test_container.client_repository()


@pytest.fixture
def document_repository(test_container):
    """Get document repository from container."""
    return test_container.document_repository()


@pytest.fixture
def document_chunk_repository(test_container):
    """Get document chunk repository from container."""
    return test_container.document_chunk_repository()


@pytest.fixture
def chunks_search_repository(test_container):
    """Get chunks search repository from container."""
    return test_container.chunks_search_repository()


@pytest.fixture
def client_search_repository(test_container):
    """Get client search repository from container."""
    return test_container.client_search_repository()


# =============================================================================
# Common service fixtures (available to all test directories)
# =============================================================================

@pytest.fixture
def unit_of_work_fixture(test_container):
    """Get unit of work from container."""
    return test_container.unit_of_work()


@pytest.fixture
def document_service(test_container):
    """Get document service from container."""
    return test_container.document_service()


@pytest.fixture
def client_service(test_container):
    """Get client service from container."""
    return test_container.client_service()


@pytest.fixture
def embedding_service(test_container):
    """Get embedding service from container."""
    return test_container.embedding_service()


@pytest.fixture
def reranker_service(test_container):
    """Get reranker service from container."""
    return test_container.reranker_service()


@pytest.fixture
def chunk_search_service(test_container):
    """Get document chunk search service from container."""
    return test_container.document_chunk_search_service()


@pytest.fixture
def document_search_service(test_container):
    """Get document search service from container."""
    return test_container.document_search_service()


@pytest.fixture
def client_search_service(test_container):
    """Get client search service from container."""
    return test_container.client_search_service()


@pytest.fixture
def search_service(test_container):
    """Get unified search service WITH reranking from container."""
    return test_container.search_service()


@pytest.fixture
def search_service_no_rerank(test_container):
    """Get unified search service WITHOUT reranking from container."""
    return test_container.search_service_no_rerank()


@pytest.fixture
def chunking_service(test_container):
    """Get chunking service from container."""
    return test_container.chunking_service()


@pytest.fixture
def tokenizer(test_container):
    """Get the tokenizer singleton from container."""
    return test_container.tokenizer()


@pytest.fixture
def text_splitter(test_container):
    """Get the text splitter singleton from container."""
    return test_container.text_splitter()


@pytest.fixture
def tokenizer_model(test_container):
    """Get the tokenizer model name from container config for custom chunking strategies."""
    return test_container.config().embedding.model_name
