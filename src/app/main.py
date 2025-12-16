import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI
from sqlalchemy import text

from src.app.api.v1 import clients, documents, search
from src.app.containers import Container
from src.app.logging import configure_logging

# Configure logging at module load time
configure_logging()

logger = logging.getLogger(__name__)

# Type alias for lifespan context manager
LifespanType = Callable[[FastAPI], AsyncIterator[None]]


@asynccontextmanager
async def default_lifespan(app: FastAPI):
    """Default application lifespan manager - initializes database on startup."""
    container: Container = app.state.container
    logger.info("Starting Nevis API...")

    # Initialize database tables on startup
    db = container.database()

    async with db._engine.begin() as conn:
        # Enable pgvector extension
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # Enable pg_trgm extension for fuzzy search
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))

        # Create all tables
        from src.shared.database.database import Base
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Database initialized successfully")

    yield

    logger.info("Shutting down Nevis API...")
    await db._engine.dispose()

def create_app(container: Container) -> FastAPI:
    """
    Create and configure FastAPI application.

    Args:
        container: Optional DI container. If not provided, creates a new one.
        lifespan: Optional lifespan context manager. If not provided, uses default_lifespan.

    Returns:
        Configured FastAPI application.
    """
    container.wire(modules=[
        "src.app.api.v1.clients",
        "src.app.api.v1.documents",
        "src.app.api.v1.search",
    ])

    config = container.config()

    app = FastAPI(
        title=config.app_name,
        version=config.app_version,
        lifespan=default_lifespan,
    )

    # Attach container to app state for access in lifespan and routes
    app.state.container = container

    # Include routers
    app.include_router(clients.router, prefix="/api/v1")
    app.include_router(documents.router, prefix="/api/v1")
    app.include_router(search.router, prefix="/api/v1")

    @app.get("/")
    async def root():
        return {"message": "Welcome to Nevis API"}

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    return app

container = Container()
app = create_app(container=container)
