import os
import threading
from pathlib import Path
from typing import AsyncGenerator
from fastapi import Depends
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.app.core.domain.models import Client, Document, DocumentChunk
from src.app.core.services.client_service import ClientService
from src.app.core.services.document_service import DocumentService
from src.app.core.services.document_processor import DocumentProcessor
from src.app.core.services.chunking import RecursiveChunkingStrategy
from src.app.core.services.embedding import SentenceTransformerEmbedding
from src.app.core.services.reranker import CrossEncoderReranker
from src.app.core.services.chunks_search_service import DocumentChunkSearchService
from src.app.core.services.document_search_service import DocumentSearchService
from src.app.core.services.client_search_service import ClientSearchService
from src.app.core.services.search_service import SearchService
from src.shared.database.database import Database, DatabaseSettings
from src.shared.database.unit_of_work import UnitOfWork
from src.shared.database.entity_mapper import EntityMapper
from src.shared.blob_storage.s3_blober import S3BlobStorage, S3BlobStorageSettings

from src.app.infrastructure.client_repository import ClientRepository
from src.app.infrastructure.document_repository import DocumentRepository
from src.app.infrastructure.document_chunk_repository import DocumentChunkRepository
from src.app.infrastructure.document_search_repository import DocumentSearchRepository
from src.app.infrastructure.client_search_repository import ClientSearchRepository
from src.app.infrastructure.mappers.client_mapper import ClientMapper
from src.app.infrastructure.mappers.document_mapper import DocumentMapper
from src.app.infrastructure.mappers.document_chunk_mapper import DocumentChunkMapper

from src.app.config import get_settings


# Global singleton instances for ML models (loaded once, reused across all requests)
_sentence_transformer_model: SentenceTransformer | None = None
_cross_encoder_model: CrossEncoder | None = None

# Thread locks to prevent concurrent model initialization (double-checked locking pattern)
_sentence_transformer_lock = threading.Lock()
_cross_encoder_lock = threading.Lock()


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
    settings = get_settings()
    return RecursiveChunkingStrategy(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )


def get_sentence_transformer_model() -> SentenceTransformer:
    """
    Get SentenceTransformer model instance (singleton with thread-safe initialization).

    Uses double-checked locking pattern to ensure:
    1. Only one thread initializes the model (prevents concurrent initialization issues)
    2. Fast path for already-loaded model (no lock needed)
    3. Memory efficient (~400MB shared across all requests)
    4. Performance optimized (model loading happens only once)

    Checks for local model first, otherwise downloads from HuggingFace.
    """
    global _sentence_transformer_model

    # Fast path: return cached model if already loaded (no lock needed)
    if _sentence_transformer_model is not None:
        return _sentence_transformer_model

    # Slow path: acquire lock and double-check before loading
    with _sentence_transformer_lock:
        # Double-check after acquiring lock (another thread may have loaded it)
        if _sentence_transformer_model is not None:
            return _sentence_transformer_model

        # Load model for the first time (only one thread will reach here)
        settings = get_settings()
        model_name = settings.embedding_model_name

        # Try to find the model in the models directory
        base_dir = Path(__file__).parent.parent.parent  # Go up to project root
        models_dir = base_dir / "models"
        model_path = models_dir / model_name

        # If local model exists, use it; otherwise download from HuggingFace
        if model_path.exists() and os.path.isdir(model_path):
            _sentence_transformer_model = SentenceTransformer(
                str(model_path),
                device='cpu'
            )
        else:
            _sentence_transformer_model = SentenceTransformer(
                model_name,
                device='cpu'
            )

        return _sentence_transformer_model


def get_cross_encoder_model() -> CrossEncoder:
    """
    Get CrossEncoder model instance for reranking (singleton with thread-safe initialization).

    Uses double-checked locking pattern to ensure:
    1. Only one thread initializes the model (prevents concurrent initialization issues)
    2. Fast path for already-loaded model (no lock needed)
    3. Memory efficient (~400MB shared across all requests)
    4. Performance optimized (model loading happens only once)

    Checks for local model first, otherwise downloads from HuggingFace.
    """
    global _cross_encoder_model

    # Fast path: return cached model if already loaded (no lock needed)
    if _cross_encoder_model is not None:
        return _cross_encoder_model

    # Slow path: acquire lock and double-check before loading
    with _cross_encoder_lock:
        # Double-check after acquiring lock (another thread may have loaded it)
        if _cross_encoder_model is not None:
            return _cross_encoder_model

        # Load model for the first time (only one thread will reach here)
        settings = get_settings()
        model_name = settings.reranker_model_name

        # Try to find the model in the models directory
        base_dir = Path(__file__).parent.parent.parent  # Go up to project root
        models_dir = base_dir / "models"
        model_path = models_dir / model_name

        # If local model exists, use it; otherwise download from HuggingFace
        if model_path.exists() and os.path.isdir(model_path):
            _cross_encoder_model = CrossEncoder(str(model_path))
        else:
            _cross_encoder_model = CrossEncoder(model_name)

        return _cross_encoder_model


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


def get_reranker_service(
    cross_encoder_model: CrossEncoder = Depends(get_cross_encoder_model)
) -> CrossEncoderReranker:
    """Get reranker service instance."""
    return CrossEncoderReranker(cross_encoder_model)


def get_document_search_repository(
    db: Database = Depends(get_database),
    mapper: DocumentChunkMapper = Depends(get_document_chunk_mapper)
) -> DocumentSearchRepository:
    """Get document search repository instance."""
    return DocumentSearchRepository(db, mapper)


def get_client_search_repository(
    db: Database = Depends(get_database),
    mapper: ClientMapper = Depends(get_client_mapper)
) -> ClientSearchRepository:
    """Get client search repository instance."""
    return ClientSearchRepository(db, mapper)


def get_document_chunk_search_service(
    embedding_service: SentenceTransformerEmbedding = Depends(get_embedding_service),
    search_repository: DocumentSearchRepository = Depends(get_document_search_repository),
    reranker_service: CrossEncoderReranker = Depends(get_reranker_service)
) -> DocumentChunkSearchService:
    """Get document chunk search service instance."""
    return DocumentChunkSearchService(
        embedding_service=embedding_service,
        search_repository=search_repository,
        reranker_service=reranker_service,
    )


def get_document_search_service(
    chunk_search_service: DocumentChunkSearchService = Depends(get_document_chunk_search_service),
    document_repository: DocumentRepository = Depends(get_document_repository)
) -> DocumentSearchService:
    """Get document search service instance."""
    return DocumentSearchService(
        chunk_search_service=chunk_search_service,
        document_repository=document_repository,
    )


def get_client_search_service(
    search_repository: ClientSearchRepository = Depends(get_client_search_repository)
) -> ClientSearchService:
    """Get client search service instance."""
    return ClientSearchService(search_repository=search_repository)


def get_search_service(
    client_search_service: ClientSearchService = Depends(get_client_search_service),
    document_search_service: DocumentSearchService = Depends(get_document_search_service)
) -> SearchService:
    """Get unified search service instance."""
    return SearchService(
        client_search_service=client_search_service,
        document_search_service=document_search_service,
    )
