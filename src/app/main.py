import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from src.app.api.v1 import clients, documents, search
from src.app.config import get_settings
from src.app.logging import configure_logging
from src.shared.database.database import Database, DatabaseSettings

# Configure logging at module load time
configure_logging()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Nevis API...")
    settings = get_settings()

    # Initialize database tables on startup
    logger.info("Initializing database...")
    db_settings = DatabaseSettings(db_url=settings.database_url)
    db = Database(db_settings)

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


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan
    )

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
