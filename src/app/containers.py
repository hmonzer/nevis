"""Dependency injection container using dependency-injector library."""
from dependency_injector import containers, providers

from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer

from src.app.config import Settings
from src.shared.database.database import Database, DatabaseSettings
from src.shared.database.unit_of_work import UnitOfWork
from src.shared.database.entity_mapper import EntityMapper
from src.shared.blob_storage.s3_blober import S3BlobStorage, S3BlobStorageSettings

from src.app.infrastructure.mappers.client_mapper import ClientMapper
from src.app.infrastructure.mappers.document_mapper import DocumentMapper
from src.app.infrastructure.mappers.document_chunk_mapper import DocumentChunkMapper

from src.app.infrastructure.client_repository import ClientRepository
from src.app.infrastructure.document_repository import DocumentRepository
from src.app.infrastructure.document_chunk_repository import DocumentChunkRepository
from src.app.infrastructure.chunks_search_repository import ChunksRepositorySearch
from src.app.infrastructure.client_search_repository import ClientSearchRepository

from src.app.core.services.client_service import ClientService
from src.app.core.services.document_service import DocumentService
from src.app.core.services.embedding import SentenceTransformerEmbedding
from src.app.core.services.reranker import CrossEncoderReranker
from src.app.core.services.chunking import RecursiveChunkingStrategy
from src.app.core.services.document_processor import DocumentProcessor
from src.app.core.services.chunks_search_service import DocumentChunkSearchService
from src.app.core.services.document_search_service import DocumentSearchService
from src.app.core.services.client_search_service import ClientSearchService
from src.app.core.services.search_service import SearchService
from src.app.core.services.rrf import ReciprocalRankFusion
from src.app.core.services.summarization import (
    SummarizationService,
    ClaudeSummarizationService,
    GeminiSummarizationService,
)

from src.app.core.domain.models import Client, Document, DocumentChunk


def create_entity_mapper(
    client_mapper: ClientMapper,
    document_mapper: DocumentMapper,
    document_chunk_mapper: DocumentChunkMapper,
) -> EntityMapper:
    """Factory function to create EntityMapper with proper mappings."""
    return EntityMapper(
        entity_mappings={
            Client: client_mapper.to_entity,
            Document: document_mapper.to_entity,
            DocumentChunk: document_chunk_mapper.to_entity,
        }
    )


def create_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Factory function to create HuggingFace tokenizer.

    The tokenizer is used for accurate token counting during text chunking,
    ensuring chunks respect the embedding model's context window.
    """
    return AutoTokenizer.from_pretrained(model_name)


def create_text_splitter(
    tokenizer: AutoTokenizer,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str] | None = None,
) -> TextSplitter:
    """
    Factory function to create text splitter with tokenizer-based splitting.

    Uses the HuggingFace tokenizer to count tokens accurately, ensuring
    chunks are sized by tokens rather than characters.
    """
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,  # type: ignore[arg-type]  # AutoTokenizer is compatible with PreTrainedTokenizerBase
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators or ["\n\n", "\n", ". ", " ", ""],
    )


def create_summarization_service(config: Settings) -> SummarizationService | None:
    """
    Factory function to create summarization service based on configuration.

    Returns the appropriate LLM-based summarization service if:
    1. Summarization is enabled in settings
    2. The required API key is available

    If summarization is disabled or API key is missing, returns None.
    """
    if not config.summarization_enabled:
        return None

    if config.summarization_provider == "claude" and config.anthropic_api_key:
        return ClaudeSummarizationService(
            api_key=config.anthropic_api_key,
            model=config.claude_model,
        )

    if config.summarization_provider == "gemini" and config.google_api_key:
        return GeminiSummarizationService(
            api_key=config.google_api_key,
            model=config.gemini_model,
        )

    # No valid configuration - return None (summarization disabled)
    return None


class Container(containers.DeclarativeContainer):
    """Main application dependency injection container."""

    wiring_config = containers.WiringConfiguration(
        modules=[
            "src.app.api.v1.clients",
            "src.app.api.v1.documents",
            "src.app.api.v1.search",
        ]
    )

    # =========================================================================
    # CONFIGURATION - Singleton (loaded once, cached)
    # =========================================================================
    config = providers.Singleton(Settings)

    # =========================================================================
    # SINGLETONS - ML Models (thread-safe, loaded once)
    # These are heavy models (~400MB each) that should only be loaded once.
    # dependency-injector handles thread safety automatically.
    # =========================================================================
    sentence_transformer_model = providers.Singleton(
        SentenceTransformer,
        model_name_or_path=config.provided.embedding_model_name,
        device="cpu",
    )

    cross_encoder_model = providers.Singleton(
        CrossEncoder,
        model_name=config.provided.chunk_reranker_model_name,
    )

    # =========================================================================
    # SINGLETONS - Stateless Mappers (reusable across all requests)
    # =========================================================================
    client_mapper = providers.Singleton(ClientMapper)
    document_mapper = providers.Singleton(DocumentMapper)
    document_chunk_mapper = providers.Singleton(DocumentChunkMapper)

    # =========================================================================
    # SINGLETONS - Composed Utilities
    # =========================================================================
    entity_mapper = providers.Singleton(
        create_entity_mapper,
        client_mapper=client_mapper,
        document_mapper=document_mapper,
        document_chunk_mapper=document_chunk_mapper,
    )

    rrf = providers.Singleton(ReciprocalRankFusion, k=60)

    # =========================================================================
    # SINGLETONS - Text Processing (tokenizer and splitter for chunking)
    # The tokenizer is loaded once and reused. It's stateless and thread-safe.
    # The text splitter is configured once with chunk settings and reused.
    # =========================================================================
    tokenizer = providers.Singleton(
        create_tokenizer,
        model_name=config.provided.embedding_model_name,
    )

    text_splitter = providers.Singleton(
        create_text_splitter,
        tokenizer=tokenizer,
        chunk_size=config.provided.chunk_size,
        chunk_overlap=config.provided.chunk_overlap,
    )

    chunking_service = providers.Singleton(
        RecursiveChunkingStrategy,
        splitter=text_splitter,
    )

    # =========================================================================
    # SINGLETON - Summarization Service (optional, based on config)
    # =========================================================================
    summarization_service = providers.Singleton(
        create_summarization_service,
        config=config,
    )

    # =========================================================================
    # SINGLETON - Database (shared connection pool)
    # =========================================================================
    database_settings = providers.Singleton(
        DatabaseSettings,
        db_url=config.provided.database_url,
    )

    database = providers.Singleton(
        Database,
        db_settings=database_settings,
    )

    # =========================================================================
    # SINGLETON - S3 Storage
    # =========================================================================
    s3_storage_settings = providers.Singleton(
        S3BlobStorageSettings,
        bucket_name=config.provided.s3_bucket_name,
        endpoint_url=config.provided.s3_endpoint_url,
        region_name=config.provided.aws_region,
        aws_access_key_id=config.provided.aws_access_key_id,
        aws_secret_access_key=config.provided.aws_secret_access_key,
    )

    s3_storage = providers.Singleton(
        S3BlobStorage,
        settings=s3_storage_settings,
    )

    # =========================================================================
    # FACTORIES - Repositories (per-request, share database singleton)
    # =========================================================================
    client_repository = providers.Factory(
        ClientRepository,
        db=database,
        mapper=client_mapper,
    )

    document_repository = providers.Factory(
        DocumentRepository,
        db=database,
        mapper=document_mapper,
    )

    document_chunk_repository = providers.Factory(
        DocumentChunkRepository,
        db=database,
        mapper=document_chunk_mapper,
    )

    chunks_search_repository = providers.Factory(
        ChunksRepositorySearch,
        db=database,
        mapper=document_chunk_mapper,
    )

    client_search_repository = providers.Factory(
        ClientSearchRepository,
        db=database,
        mapper=client_mapper,
    )

    # =========================================================================
    # FACTORY - Unit of Work (per-request)
    # =========================================================================
    unit_of_work = providers.Factory(
        UnitOfWork,
        db=database,
        entity_mapper=entity_mapper,
    )

    # =========================================================================
    # FACTORIES - Services
    # =========================================================================
    embedding_service = providers.Factory(
        SentenceTransformerEmbedding,
        model=sentence_transformer_model,
    )

    reranker_service = providers.Factory(
        CrossEncoderReranker,
        model=cross_encoder_model,
    )

    document_processor = providers.Factory(
        DocumentProcessor,
        chunking_strategy=chunking_service,
        embedding_service=embedding_service,
        summarization_service=summarization_service,
    )

    client_service = providers.Factory(
        ClientService,
        repository=client_repository,
        unit_of_work=unit_of_work,
    )

    document_service = providers.Factory(
        DocumentService,
        client_repository=client_repository,
        document_repository=document_repository,
        unit_of_work=unit_of_work,
        blob_storage=s3_storage,
        document_processor=document_processor,
    )

    document_chunk_search_service = providers.Factory(
        DocumentChunkSearchService,
        embedding_service=embedding_service,
        search_repository=chunks_search_repository,
        rrf=rrf,
        reranker_service=reranker_service,
        reranker_score_threshold=config.provided.chunk_reranker_score_threshold,
    )

    # Variant without reranking (for testing/comparison)
    document_chunk_search_service_no_rerank = providers.Factory(
        DocumentChunkSearchService,
        embedding_service=embedding_service,
        search_repository=chunks_search_repository,
        rrf=rrf,
        reranker_service=None,
    )

    document_search_service = providers.Factory(
        DocumentSearchService,
        chunk_search_service=document_chunk_search_service,
        document_repository=document_repository,
    )

    # Variant without reranking (for testing/comparison)
    document_search_service_no_rerank = providers.Factory(
        DocumentSearchService,
        chunk_search_service=document_chunk_search_service_no_rerank,
        document_repository=document_repository,
    )

    client_search_service = providers.Factory(
        ClientSearchService,
        search_repository=client_search_repository,
    )

    search_service = providers.Factory(
        SearchService,
        client_search_service=client_search_service,
        document_search_service=document_search_service,
    )

    # Variant without reranking (for testing/comparison)
    search_service_no_rerank = providers.Factory(
        SearchService,
        client_search_service=client_search_service,
        document_search_service=document_search_service_no_rerank,
    )
