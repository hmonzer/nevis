import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from src.app.api.v1 import clients, documents, search
from src.app.containers import Container
from src.app.logging import configure_logging

# Configure logging at module load time
configure_logging()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    container: Container = app.state.container

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

    await db._engine.dispose()
    logger.info("Database initialized successfully")

    yield

    logger.info("Shutting down Nevis API...")
    # Cleanup on shutdown
    await db._engine.dispose()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    # Initialize the DI container
    container = Container()
    container.wire(modules=[
        "src.app.api.v1.clients",
        "src.app.api.v1.documents",
        "src.app.api.v1.search",
    ])

    config = container.config()

    app = FastAPI(
        title=config.app_name,
        version=config.app_version,
        lifespan=lifespan
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


app = create_app()
