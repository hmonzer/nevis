import os
from pathlib import Path
from typing import AsyncGenerator
from fastapi import Depends
from sentence_transformers import SentenceTransformer

from src.app.core.domain.models import Client, Document, DocumentChunk
from src.app.core.services.client_service import ClientService
from src.app.core.services.document_service import DocumentService
from src.app.core.services.document_processor import DocumentProcessor
from src.app.core.services.chunking import RecursiveChunkingStrategy
from src.app.core.services.embedding import SentenceTransformerEmbedding
from src.shared.database.database import Database, DatabaseSettings
from src.shared.database.unit_of_work import UnitOfWork
from src.shared.database.entity_mapper import EntityMapper
from src.shared.blob_storage.s3_blober import S3BlobStorage, S3BlobStorageSettings

from src.app.infrastructure.client_repository import ClientRepository
from src.app.infrastructure.document_repository import DocumentRepository
from src.app.infrastructure.document_chunk_repository import DocumentChunkRepository
from src.app.infrastructure.mappers.client_mapper import ClientMapper
from src.app.infrastructure.mappers.document_mapper import DocumentMapper
from src.app.infrastructure.mappers.document_chunk_mapper import DocumentChunkMapper

from src.app.config import get_settings


async def get_database() -> AsyncGenerator[Database, None]:
    """Get database instance."""
    settings = get_settings()
    db_settings = DatabaseSettings(db_url=settings.database_url)
    db = Database(db_settings)
    try:
        yield db
    finally:
        await db._engine.dispose()


def get_client_mapper() -> ClientMapper:
    """Get a Client mapper instance."""
    return ClientMapper()


def get_client_repository(
    db: Database = Depends(get_database),
    mapper: ClientMapper = Depends(get_client_mapper)
) -> ClientRepository:
    """Get Client repository instance."""
    return ClientRepository(db, mapper)


def get_document_mapper() -> DocumentMapper:
    """Get Document mapper instance."""
    return DocumentMapper()


def get_document_chunk_mapper() -> DocumentChunkMapper:
    """Get DocumentChunk mapper instance."""
    return DocumentChunkMapper()


def get_document_repository(
    db: Database = Depends(get_database),
    mapper: DocumentMapper = Depends(get_document_mapper)
) -> DocumentRepository:
    """Get Document repository instance."""
    return DocumentRepository(db, mapper)


def get_document_chunk_repository(
    db: Database = Depends(get_database),
    mapper: DocumentChunkMapper = Depends(get_document_chunk_mapper)
) -> DocumentChunkRepository:
    """Get DocumentChunk repository instance."""
    return DocumentChunkRepository(db, mapper)


def get_entity_mapper(
    client_mapper: ClientMapper = Depends(get_client_mapper),
    document_mapper: DocumentMapper = Depends(get_document_mapper),
    document_chunk_mapper: DocumentChunkMapper = Depends(get_document_chunk_mapper)
) -> EntityMapper:
    """Get entity mapper with all model mappings."""
    return EntityMapper(
        entity_mappings={
            Client: client_mapper.to_entity,
            Document: document_mapper.to_entity,
            DocumentChunk: document_chunk_mapper.to_entity,
        }
    )


def get_unit_of_work(
    db: Database = Depends(get_database),
    entity_mapper: EntityMapper = Depends(get_entity_mapper)
) -> UnitOfWork:
    """Get Unit of Work instance."""
    return UnitOfWork(db, entity_mapper)


def get_client_service(
    repository: ClientRepository = Depends(get_client_repository),
    unit_of_work: UnitOfWork = Depends(get_unit_of_work)
) -> ClientService:
    """Get Client service instance."""
    return ClientService(repository, unit_of_work)


def get_s3_storage() -> S3BlobStorage:
    """Get S3 blob storage instance."""
    settings = get_settings()
    storage_settings = S3BlobStorageSettings(
        bucket_name=settings.s3_bucket_name,
        endpoint_url=settings.s3_endpoint_url,
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )
    return S3BlobStorage(storage_settings)


def get_chunking_service() -> RecursiveChunkingStrategy:
    """Get chunking service instance with default recursive strategy."""
    return RecursiveChunkingStrategy(
        chunk_size=1000,
        chunk_overlap=200
    )


def get_sentence_transformer_model() -> SentenceTransformer:
    """
    Get SentenceTransformer model instance.

    Checks for local model first, otherwise downloads from HuggingFace.
    """
    # model_name = "all-MiniLM-L6-v2"
    model_name = "google/embeddinggemma-300m"

    # Try to find the model in the models directory
    base_dir = Path(__file__).parent.parent.parent  # Go up to project root
    models_dir = base_dir / "models"
    model_path = models_dir / model_name

    # If local model exists, use it; otherwise download from HuggingFace
    if model_path.exists() and os.path.isdir(model_path):
        return SentenceTransformer(str(model_path))
    else:
        return SentenceTransformer(model_name)


def get_embedding_service(
    model: SentenceTransformer = Depends(get_sentence_transformer_model)
) -> SentenceTransformerEmbedding:
    """Get embedding service instance with injected SentenceTransformer model."""
    return SentenceTransformerEmbedding(model)


def get_document_processor(
    chunking_service: RecursiveChunkingStrategy = Depends(get_chunking_service),
    embedding_service: SentenceTransformerEmbedding = Depends(get_embedding_service),
) -> DocumentProcessor:
    """Get Document processor instance."""
    return DocumentProcessor(
        chunking_strategy=chunking_service,
        embedding_service=embedding_service,
    )


def get_document_service(
    client_repository: ClientRepository = Depends(get_client_repository),
    document_repository: DocumentRepository = Depends(get_document_repository),
    unit_of_work: UnitOfWork = Depends(get_unit_of_work),
    blob_storage: S3BlobStorage = Depends(get_s3_storage),
    document_processor: DocumentProcessor = Depends(get_document_processor),
) -> DocumentService:
    """Get Document service instance."""
    return DocumentService(
        client_repository=client_repository,
        document_repository=document_repository,
        unit_of_work=unit_of_work,
        blob_storage=blob_storage,
        document_processor=document_processor,
    )
