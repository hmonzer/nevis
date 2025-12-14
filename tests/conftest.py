"""Shared test fixtures and utilities for all tests."""
import asyncio
import os

import pytest
import pytest_asyncio
from testcontainers.postgres import PostgresContainer

from src.shared.database.database import Database, Base
from src.shared.database.database_settings import DatabaseSettings


@pytest.fixture(scope="module")
def postgres_container():
    """Start a PostgreSQL container for testing. Module-scoped for reuse."""
    with PostgresContainer("postgres:16-alpine") as postgres:
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
def test_settings_override(async_db_url):
    """
    Centralized settings override for all test configurations.
    Module-scoped to set up environment once per test module.

    This fixture manages all environment variable overrides needed for testing,
    providing a single place to configure test settings.
    """

    os.environ["DATABASE_URL"] = async_db_url

    # # Clear settings cache to force reload with new env vars
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
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield db
