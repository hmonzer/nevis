"""Tests for DocumentSearchRepository vector search functionality."""
from uuid import uuid4

import pytest
import pytest_asyncio

from src.app.core.domain.models import Document, DocumentChunk, DocumentStatus
from src.shared.database.unit_of_work import UnitOfWork
from src.shared.database.entity_mapper import EntityMapper

from src.app.infrastructure.chunks_search_repository import ChunksRepositorySearch
from src.app.infrastructure.mappers.document_chunk_mapper import DocumentChunkMapper
from src.app.infrastructure.mappers.document_mapper import DocumentMapper
from src.app.infrastructure.mappers.client_mapper import ClientMapper
from src.app.core.domain.models import Client
from pydantic.v1 import EmailStr


@pytest_asyncio.fixture
async def chunk_search_repository(clean_database):
    """Create a document search repository."""
    return ChunksRepositorySearch(clean_database, DocumentChunkMapper())


@pytest_asyncio.fixture
async def unit_of_work(clean_database):
    """Create a unit of work for test data setup."""
    entity_mapper = EntityMapper(
        entity_mappings={
            Client: ClientMapper().to_entity,
            Document: DocumentMapper().to_entity,
            DocumentChunk: DocumentChunkMapper().to_entity,
        }
    )
    return UnitOfWork(clean_database, entity_mapper)


@pytest.mark.asyncio
async def test_search_by_vector_returns_similar_chunks(chunk_search_repository, unit_of_work):
    """Test that vector search returns chunks ordered by similarity."""
    # Arrange - Create a client and document
    client = Client(
        id=uuid4(),
        first_name="Test",
        last_name="User",
        email=EmailStr("test@example.com")
    )

    document = Document(
        id=uuid4(),
        client_id=client.id,
        title="Test Document",
        s3_key="test/document.pdf",
        status=DocumentStatus.PROCESSED
    )

    # Create chunks with different embeddings
    # Embedding 1: Opposite direction - negative values (low similarity to positive query)
    chunk1 = DocumentChunk(
        id=uuid4(),
        document_id=document.id,
        chunk_index=0,
        chunk_content="This is chunk 1",
        embedding=[-1.0] * 384  # Opposite direction = low similarity
    )

    # Embedding 2: Same direction as query (high similarity)
    chunk2 = DocumentChunk(
        id=uuid4(),
        document_id=document.id,
        chunk_index=1,
        chunk_content="This is chunk 2",
        embedding=[1.0] * 384  # Same direction = high similarity
    )

    # Embedding 3: Mixed values (medium similarity)
    chunk3 = DocumentChunk(
        id=uuid4(),
        document_id=document.id,
        chunk_index=2,
        chunk_content="This is chunk 3",
        embedding=[0.5 if i % 2 == 0 else -0.5 for i in range(384)]  # Mixed = medium similarity
    )

    # Persist all entities
    async with unit_of_work:
        unit_of_work.add(client)
        unit_of_work.add(document)
        unit_of_work.add(chunk1)
        unit_of_work.add(chunk2)
        unit_of_work.add(chunk3)

    # Act - Search with a query vector of all 1.0
    query_vector = [1.0] * 384
    results = await chunk_search_repository.search_by_vector(query_vector, limit=10)

    # Assert - Results should be ordered by similarity (chunk2, chunk3, chunk1)
    assert len(results) == 3
    assert results[0].item.id == chunk2.id  # Most similar
    assert results[1].item.id == chunk3.id  # Medium similarity
    assert results[2].item.id == chunk1.id  # Least similar

    # Verify scores are in descending order
    assert results[0].value > results[1].value > results[2].value

    # Verify scores are in valid range for cosine similarity [-1.0, 1.0]
    for result in results:
        assert -1.0 <= result.value <= 1.0, "Cosine similarity scores should be in range [-1.0, 1.0]"


@pytest.mark.asyncio
async def test_search_by_vector_respects_limit(chunk_search_repository, unit_of_work):
    """Test that the limit parameter restricts the number of results."""
    # Arrange - Create a client, document, and multiple chunks
    client = Client(
        id=uuid4(),
        first_name="Test",
        last_name="User",
        email=EmailStr("test@example.com")
    )

    document = Document(
        id=uuid4(),
        client_id=client.id,
        title="Test Document",
        s3_key="test/document.pdf",
        status=DocumentStatus.PROCESSED
    )

    # Create 5 chunks with embeddings
    chunks = []
    async with unit_of_work:
        unit_of_work.add(client)
        unit_of_work.add(document)

        for i in range(5):
            chunk = DocumentChunk(
                id=uuid4(),
                document_id=document.id,
                chunk_index=i,
                chunk_content=f"This is chunk {i}",
                embedding=[float(i) / 10] * 384
            )
            chunks.append(chunk)
            unit_of_work.add(chunk)

    # Act - Search with limit=3
    query_vector = [1.0] * 384
    results = await chunk_search_repository.search_by_vector(query_vector, limit=3)

    # Assert - Should only return 3 results
    assert len(results) == 3


@pytest.mark.asyncio
async def test_search_by_vector_with_similarity_threshold(chunk_search_repository, unit_of_work):
    """Test that similarity_threshold filters out low-scoring results."""
    # Arrange - Create a client, document, and chunks
    client = Client(
        id=uuid4(),
        first_name="Test",
        last_name="User",
        email=EmailStr("test@example.com")
    )

    document = Document(
        id=uuid4(),
        client_id=client.id,
        title="Test Document",
        s3_key="test/document.pdf",
        status=DocumentStatus.PROCESSED
    )

    # Create chunks with varying similarity to query vector [1.0, 1.0, ...]
    # High similarity chunk - same direction as query
    high_sim_chunk = DocumentChunk(
        id=uuid4(),
        document_id=document.id,
        chunk_index=0,
        chunk_content="High similarity chunk",
        embedding=[1.0] * 384  # Same direction = very high similarity (score ~ 1.0)
    )

    # Low similarity chunk - opposite direction from query
    low_sim_chunk = DocumentChunk(
        id=uuid4(),
        document_id=document.id,
        chunk_index=1,
        chunk_content="Low similarity chunk",
        embedding=[-1.0] * 384  # Opposite direction = very low similarity (score ~ 0.0)
    )

    async with unit_of_work:
        unit_of_work.add(client)
        unit_of_work.add(document)
        unit_of_work.add(high_sim_chunk)
        unit_of_work.add(low_sim_chunk)

    # Act - Search with high similarity threshold (0.5)
    # This should filter out the low similarity chunk (score ~ -1.0) but keep the high one (score ~ 1.0)
    query_vector = [1.0] * 384
    results = await chunk_search_repository.search_by_vector(
        query_vector,
        limit=10,
        similarity_threshold=0.5
    )

    # Assert - Should only return the high similarity chunk
    assert len(results) == 1
    assert results[0].item.id == high_sim_chunk.id
    assert results[0].value >= 0.5


@pytest.mark.asyncio
async def test_search_by_vector_excludes_chunks_without_embeddings(chunk_search_repository, unit_of_work):
    """Test that chunks without embeddings are excluded from search results."""
    # Arrange - Create chunks with and without embeddings
    client = Client(
        id=uuid4(),
        first_name="Test",
        last_name="User",
        email=EmailStr("test@example.com")
    )

    document = Document(
        id=uuid4(),
        client_id=client.id,
        title="Test Document",
        s3_key="test/document.pdf",
        status=DocumentStatus.PROCESSED
    )

    # Chunk with embedding
    chunk_with_embedding = DocumentChunk(
        id=uuid4(),
        document_id=document.id,
        chunk_index=0,
        chunk_content="Chunk with embedding",
        embedding=[0.5] * 384
    )

    # Chunk without embedding
    chunk_without_embedding = DocumentChunk(
        id=uuid4(),
        document_id=document.id,
        chunk_index=1,
        chunk_content="Chunk without embedding",
        embedding=None
    )

    async with unit_of_work:
        unit_of_work.add(client)
        unit_of_work.add(document)
        unit_of_work.add(chunk_with_embedding)
        unit_of_work.add(chunk_without_embedding)

    # Act - Search for chunks
    query_vector = [1.0] * 384
    results = await chunk_search_repository.search_by_vector(query_vector, limit=10)

    # Assert - Should only return the chunk with embedding
    assert len(results) == 1
    assert results[0].item.id == chunk_with_embedding.id


@pytest.mark.asyncio
async def test_search_by_vector_empty_query_raises_error(chunk_search_repository):
    """Test that searching with an empty query vector raises ValueError."""
    # Act & Assert
    with pytest.raises(ValueError, match="Query vector cannot be empty"):
        await chunk_search_repository.search_by_vector([], limit=10)


@pytest.mark.asyncio
async def test_search_by_vector_wrong_dimensions_raises_error(chunk_search_repository):
    """Test that searching with wrong vector dimensions raises ValueError."""
    # Act & Assert - Vector with wrong dimensions (not 384)
    with pytest.raises(ValueError, match="Query vector must be 384-dimensional"):
        await chunk_search_repository.search_by_vector([1.0] * 128, limit=10)


@pytest.mark.asyncio
async def test_search_by_vector_returns_empty_when_no_chunks(chunk_search_repository):
    """Test that searching returns empty list when no chunks exist."""
    # Act - Search with no chunks in database
    query_vector = [1.0] * 384
    results = await chunk_search_repository.search_by_vector(query_vector, limit=10)

    # Assert - Should return empty list
    assert len(results) == 0


# ============================================================================
# Keyword Search Tests
# ============================================================================

@pytest.mark.asyncio
async def test_search_by_keyword_returns_matching_chunks(chunk_search_repository, unit_of_work):
    """Test that keyword search returns chunks containing the search terms."""
    # Arrange - Create a client and document
    client = Client(
        id=uuid4(),
        first_name="Test",
        last_name="User",
        email=EmailStr("test@example.com")
    )

    document = Document(
        id=uuid4(),
        client_id=client.id,
        title="Test Document",
        s3_key="test/document.pdf",
        status=DocumentStatus.PROCESSED
    )

    # Create chunks with different content
    chunk1 = DocumentChunk(
        id=uuid4(),
        document_id=document.id,
        chunk_index=0,
        chunk_content="The quick brown fox jumps over the lazy dog",
        embedding=[0.1] * 384
    )

    chunk2 = DocumentChunk(
        id=uuid4(),
        document_id=document.id,
        chunk_index=1,
        chunk_content="Investment portfolio analysis for retirement planning",
        embedding=[0.2] * 384
    )

    chunk3 = DocumentChunk(
        id=uuid4(),
        document_id=document.id,
        chunk_index=2,
        chunk_content="The fox is a clever animal that lives in the forest",
        embedding=[0.3] * 384
    )

    async with unit_of_work:
        unit_of_work.add(client)
        unit_of_work.add(document)
        unit_of_work.add(chunk1)
        unit_of_work.add(chunk2)
        unit_of_work.add(chunk3)

    # Act - Search for "fox"
    results = await chunk_search_repository.search_by_keyword("fox", limit=10)

    # Assert - Should return chunks containing "fox"
    assert len(results) == 2
    result_ids = {r.item.id for r in results}
    assert chunk1.id in result_ids
    assert chunk3.id in result_ids
    assert chunk2.id not in result_ids  # Doesn't contain "fox"

    # Verify scores are positive
    for result in results:
        assert result.value > 0


@pytest.mark.asyncio
async def test_search_by_keyword_respects_limit(chunk_search_repository, unit_of_work):
    """Test that keyword search respects the limit parameter."""
    # Arrange - Create chunks all containing the same keyword
    client = Client(
        id=uuid4(),
        first_name="Test",
        last_name="User",
        email=EmailStr("test@example.com")
    )

    document = Document(
        id=uuid4(),
        client_id=client.id,
        title="Test Document",
        s3_key="test/document.pdf",
        status=DocumentStatus.PROCESSED
    )

    async with unit_of_work:
        unit_of_work.add(client)
        unit_of_work.add(document)

        for i in range(5):
            chunk = DocumentChunk(
                id=uuid4(),
                document_id=document.id,
                chunk_index=i,
                chunk_content=f"Investment strategy number {i} for retirement",
                embedding=[0.1] * 384
            )
            unit_of_work.add(chunk)

    # Act - Search with limit=2
    results = await chunk_search_repository.search_by_keyword("investment", limit=2)

    # Assert - Should only return 2 results
    assert len(results) == 2


@pytest.mark.asyncio
async def test_search_by_keyword_returns_empty_when_no_matches(chunk_search_repository, unit_of_work):
    """Test that keyword search returns empty list when no chunks match."""
    # Arrange - Create chunks that don't match search term
    client = Client(
        id=uuid4(),
        first_name="Test",
        last_name="User",
        email=EmailStr("test@example.com")
    )

    document = Document(
        id=uuid4(),
        client_id=client.id,
        title="Test Document",
        s3_key="test/document.pdf",
        status=DocumentStatus.PROCESSED
    )

    chunk = DocumentChunk(
        id=uuid4(),
        document_id=document.id,
        chunk_index=0,
        chunk_content="The quick brown fox jumps over the lazy dog",
        embedding=[0.1] * 384
    )

    async with unit_of_work:
        unit_of_work.add(client)
        unit_of_work.add(document)
        unit_of_work.add(chunk)

    # Act - Search for a term that doesn't exist
    results = await chunk_search_repository.search_by_keyword("cryptocurrency", limit=10)

    # Assert - Should return empty list
    assert len(results) == 0


@pytest.mark.asyncio
async def test_search_by_keyword_empty_query_raises_error(chunk_search_repository):
    """Test that searching with empty query raises ValueError."""
    # Act & Assert
    with pytest.raises(ValueError, match="Query text cannot be empty"):
        await chunk_search_repository.search_by_keyword("", limit=10)

    with pytest.raises(ValueError, match="Query text cannot be empty"):
        await chunk_search_repository.search_by_keyword("   ", limit=10)


@pytest.mark.asyncio
async def test_search_by_keyword_handles_multiple_words(chunk_search_repository, unit_of_work):
    """Test that keyword search handles multi-word queries."""
    # Arrange
    client = Client(
        id=uuid4(),
        first_name="Test",
        last_name="User",
        email=EmailStr("test@example.com")
    )

    document = Document(
        id=uuid4(),
        client_id=client.id,
        title="Test Document",
        s3_key="test/document.pdf",
        status=DocumentStatus.PROCESSED
    )

    chunk1 = DocumentChunk(
        id=uuid4(),
        document_id=document.id,
        chunk_index=0,
        chunk_content="Risk tolerance assessment for portfolio management",
        embedding=[0.1] * 384
    )

    chunk2 = DocumentChunk(
        id=uuid4(),
        document_id=document.id,
        chunk_index=1,
        chunk_content="Portfolio diversification strategies",
        embedding=[0.2] * 384
    )

    async with unit_of_work:
        unit_of_work.add(client)
        unit_of_work.add(document)
        unit_of_work.add(chunk1)
        unit_of_work.add(chunk2)

    # Act - Search for multiple words
    results = await chunk_search_repository.search_by_keyword("risk tolerance", limit=10)

    # Assert - Should return chunk containing both terms
    assert len(results) >= 1
    assert any(r.item.id == chunk1.id for r in results)


@pytest.mark.asyncio
async def test_search_by_keyword_ranks_by_relevance(chunk_search_repository, unit_of_work):
    """Test that keyword search returns results ranked by relevance."""
    # Arrange
    client = Client(
        id=uuid4(),
        first_name="Test",
        last_name="User",
        email=EmailStr("test@example.com")
    )

    document = Document(
        id=uuid4(),
        client_id=client.id,
        title="Test Document",
        s3_key="test/document.pdf",
        status=DocumentStatus.PROCESSED
    )

    # Chunk with many occurrences of search term
    chunk_high_relevance = DocumentChunk(
        id=uuid4(),
        document_id=document.id,
        chunk_index=0,
        chunk_content="Investment investment investment strategies for investment planning",
        embedding=[0.1] * 384
    )

    # Chunk with single occurrence
    chunk_low_relevance = DocumentChunk(
        id=uuid4(),
        document_id=document.id,
        chunk_index=1,
        chunk_content="An investment opportunity for clients",
        embedding=[0.2] * 384
    )

    async with unit_of_work:
        unit_of_work.add(client)
        unit_of_work.add(document)
        unit_of_work.add(chunk_high_relevance)
        unit_of_work.add(chunk_low_relevance)

    # Act
    results = await chunk_search_repository.search_by_keyword("investment", limit=10)

    # Assert - Higher relevance chunk should have higher score
    assert len(results) == 2
    # Results should be ordered by score descending
    assert results[0].value >= results[1].value
