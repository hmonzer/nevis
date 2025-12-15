"""Shared fixtures for e2e evaluation tests."""
import pytest
import pytest_asyncio
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.app.core.services.chunks_search_service import DocumentChunkSearchService
from src.app.core.services.document_search_service import DocumentSearchService
from src.app.core.services.client_search_service import ClientSearchService
from src.app.core.services.search_service import SearchService
from src.app.core.services.embedding import SentenceTransformerEmbedding
from src.app.core.services.reranker import CrossEncoderReranker
from src.app.infrastructure.document_search_repository import DocumentSearchRepository
from src.app.infrastructure.client_search_repository import ClientSearchRepository
from src.app.infrastructure.mappers.document_chunk_mapper import DocumentChunkMapper
from src.app.infrastructure.mappers.client_mapper import ClientMapper
from src.app.infrastructure.document_repository import DocumentRepository
from src.app.infrastructure.mappers.document_mapper import DocumentMapper


@pytest_asyncio.fixture(scope="module")
def sentence_transformer_model():
    """Load sentence transformer model once per module."""
    return SentenceTransformer("all-MiniLM-L6-v2")


@pytest_asyncio.fixture(scope="module")
def embedding_service(sentence_transformer_model):
    """Create embedding service."""
    return SentenceTransformerEmbedding(sentence_transformer_model)


@pytest_asyncio.fixture(scope="module")
def cross_encoder_model():
    """Load cross-encoder model once per module."""
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


@pytest_asyncio.fixture(scope="module")
def reranker_service(cross_encoder_model):
    """Create reranker service."""
    return CrossEncoderReranker(cross_encoder_model)


@pytest_asyncio.fixture
def document_search_repository(clean_database):
    """Create document search repository."""
    return DocumentSearchRepository(clean_database, DocumentChunkMapper())


@pytest_asyncio.fixture
def client_search_repository(clean_database):
    """Create client search repository."""
    return ClientSearchRepository(clean_database, ClientMapper())


@pytest_asyncio.fixture
def chunk_search_service(embedding_service, document_search_repository, reranker_service):
    """Create document chunk search service."""
    return DocumentChunkSearchService(
        embedding_service=embedding_service,
        search_repository=document_search_repository,
        reranker_service=reranker_service,
    )


@pytest_asyncio.fixture
def document_repository(clean_database):
    """Create document repository."""
    return DocumentRepository(clean_database, DocumentMapper())


@pytest_asyncio.fixture
def document_search_service(chunk_search_service, document_repository):
    """Create document search service."""
    return DocumentSearchService(
        chunk_search_service=chunk_search_service,
        document_repository=document_repository,
    )


@pytest_asyncio.fixture
def client_search_service(client_search_repository):
    """Create client search service."""
    return ClientSearchService(search_repository=client_search_repository)


@pytest_asyncio.fixture
def unified_search_service(client_search_service, document_search_service):
    """Create unified search service."""
    return SearchService(
        client_search_service=client_search_service,
        document_search_service=document_search_service,
    )
