"""Tests for DocumentSearchService."""
from uuid import uuid4

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, Mock

from src.app.core.domain.models import (
    Document,
    DocumentChunk,
    ChunkSearchResult,
    DocumentSearchResult
)
from src.app.core.services.document_search_service import DocumentSearchService
from src.app.core.services.chunks_search_service import DocumentChunkSearchService
from src.app.infrastructure.document_repository import DocumentRepository


@pytest_asyncio.fixture
async def mock_chunk_search_service():
    """Create a mock chunk search service."""
    service = AsyncMock(spec=DocumentChunkSearchService)
    return service


@pytest_asyncio.fixture
async def mock_document_repository():
    """Create a mock document repository."""
    repository = AsyncMock(spec=DocumentRepository)
    return repository


@pytest_asyncio.fixture
async def document_search_service(mock_chunk_search_service, mock_document_repository):
    """Create a document search service with mocked dependencies."""
    return DocumentSearchService(
        chunk_search_service=mock_chunk_search_service,
        document_repository=mock_document_repository
    )


@pytest.mark.asyncio
async def test_search_multiple_documents(document_search_service, mock_chunk_search_service, mock_document_repository):
    """Test searching returns multiple distinct documents."""
    # Arrange - Create documents
    doc1_id = uuid4()
    doc2_id = uuid4()
    doc3_id = uuid4()

    doc1 = Document(
        id=doc1_id,
        client_id=uuid4(),
        title="Q4 2024 Portfolio Report",
        s3_key="reports/q4-2024.pdf"
    )
    doc2 = Document(
        id=doc2_id,
        client_id=uuid4(),
        title="Investment Strategy 2025",
        s3_key="strategy/2025.pdf"
    )
    doc3 = Document(
        id=doc3_id,
        client_id=uuid4(),
        title="Risk Assessment Report",
        s3_key="risk/assessment.pdf"
    )

    # Create chunks from different documents
    chunk1 = DocumentChunk(
        id=uuid4(),
        document_id=doc1_id,
        chunk_index=0,
        chunk_content="Portfolio performance exceeded benchmarks with 12% annual return in equities."
    )
    chunk2 = DocumentChunk(
        id=uuid4(),
        document_id=doc2_id,
        chunk_index=0,
        chunk_content="Investment strategy focuses on diversified portfolio allocation across asset classes."
    )
    chunk3 = DocumentChunk(
        id=uuid4(),
        document_id=doc3_id,
        chunk_index=0,
        chunk_content="Risk assessment indicates moderate portfolio volatility within acceptable parameters."
    )

    # Mock chunk search results
    chunk_results = [
        ChunkSearchResult(chunk=chunk1, score=0.85),
        ChunkSearchResult(chunk=chunk2, score=0.78),
        ChunkSearchResult(chunk=chunk3, score=0.72),
    ]
    mock_chunk_search_service.search.return_value = chunk_results

    # Mock document repository - batch fetch
    mock_document_repository.get_by_ids.return_value = [doc1, doc2, doc3]

    # Act
    results = await document_search_service.search("portfolio performance", top_k=10)

    # Assert
    assert len(results) == 3
    assert results[0].document.id == doc1_id
    assert results[0].score == 0.85
    assert results[1].document.id == doc2_id
    assert results[1].score == 0.78
    assert results[2].document.id == doc3_id
    assert results[2].score == 0.72

    # Verify chunk search was called with correct parameters
    mock_chunk_search_service.search.assert_called_once_with(
        query="portfolio performance",
        top_k=50,  # top_k * 5
        similarity_threshold=0.5
    )

    # Verify batch fetch was used
    mock_document_repository.get_by_ids.assert_called_once()
    called_ids = mock_document_repository.get_by_ids.call_args[0][0]
    assert set(called_ids) == {doc1_id, doc2_id, doc3_id}


@pytest.mark.asyncio
async def test_search_multiple_chunks_same_document(document_search_service, mock_chunk_search_service, mock_document_repository):
    """Test that when multiple chunks from same document match, highest score is used."""
    # Arrange - Create one document
    doc_id = uuid4()
    document = Document(
        id=doc_id,
        client_id=uuid4(),
        title="Comprehensive Wealth Management Guide",
        s3_key="guides/wealth-management.pdf"
    )

    # Create multiple chunks from the same document with different scores
    chunk1 = DocumentChunk(
        id=uuid4(),
        document_id=doc_id,
        chunk_index=0,
        chunk_content="Wealth management encompasses portfolio diversification and risk mitigation strategies."
    )
    chunk2 = DocumentChunk(
        id=uuid4(),
        document_id=doc_id,
        chunk_index=1,
        chunk_content="Tax optimization strategies are crucial for high-net-worth individuals managing wealth."
    )
    chunk3 = DocumentChunk(
        id=uuid4(),
        document_id=doc_id,
        chunk_index=2,
        chunk_content="Estate planning ensures intergenerational wealth transfer efficiency."
    )

    # Mock chunk search results - different scores for chunks from same document
    chunk_results = [
        ChunkSearchResult(chunk=chunk1, score=0.92),  # Highest score
        ChunkSearchResult(chunk=chunk2, score=0.88),
        ChunkSearchResult(chunk=chunk3, score=0.75),
    ]
    mock_chunk_search_service.search.return_value = chunk_results

    # Mock document repository - batch fetch
    mock_document_repository.get_by_ids.return_value = [document]

    # Act
    results = await document_search_service.search("wealth management strategies", top_k=5)

    # Assert - Should return only one document with the highest score
    assert len(results) == 1
    assert results[0].document.id == doc_id
    assert results[0].score == 0.92  # Should use the highest score from the chunks

    # Verify batch fetch was used with single document
    mock_document_repository.get_by_ids.assert_called_once_with([doc_id])


@pytest.mark.asyncio
async def test_search_respects_top_k_limit(document_search_service, mock_chunk_search_service, mock_document_repository):
    """Test that search respects the top_k parameter when multiple documents match."""
    # Arrange - Create 5 documents but request only top 3
    documents = {}
    chunk_results = []

    for i in range(5):
        doc_id = uuid4()
        documents[doc_id] = Document(
            id=doc_id,
            client_id=uuid4(),
            title=f"Financial Report {i+1}",
            s3_key=f"reports/report-{i+1}.pdf"
        )

        chunk = DocumentChunk(
            id=uuid4(),
            document_id=doc_id,
            chunk_index=0,
            chunk_content=f"Financial analysis report {i+1} for portfolio review."
        )
        # Scores in descending order
        chunk_results.append(ChunkSearchResult(chunk=chunk, score=0.9 - (i * 0.1)))

    mock_chunk_search_service.search.return_value = chunk_results
    # Return only the top 3 documents in order
    top_3_docs = [documents[chunk_results[i].chunk.document_id] for i in range(3)]
    mock_document_repository.get_by_ids.return_value = top_3_docs

    # Act - Request only top 3
    results = await document_search_service.search("financial report", top_k=3)

    # Assert - Should return exactly 3 documents
    assert len(results) == 3
    # Verify they are the top 3 by score
    assert results[0].score == 0.9
    assert results[1].score == 0.8
    assert results[2].score == 0.7


@pytest.mark.asyncio
async def test_search_mixed_documents_and_chunks(document_search_service, mock_chunk_search_service, mock_document_repository):
    """Test search with mix of single and multiple chunks per document."""
    # Arrange
    doc1_id = uuid4()  # Will have 3 chunks
    doc2_id = uuid4()  # Will have 1 chunk
    doc3_id = uuid4()  # Will have 2 chunks

    doc1 = Document(
        id=doc1_id,
        client_id=uuid4(),
        title="Annual Investment Review 2024",
        s3_key="reviews/2024-annual.pdf"
    )
    doc2 = Document(
        id=doc2_id,
        client_id=uuid4(),
        title="Market Outlook Q1 2025",
        s3_key="outlook/q1-2025.pdf"
    )
    doc3 = Document(
        id=doc3_id,
        client_id=uuid4(),
        title="Client Portfolio Summary",
        s3_key="portfolio/client-summary.pdf"
    )

    chunk_results = [
        # Doc1 - 3 chunks, highest score is 0.95
        ChunkSearchResult(
            chunk=DocumentChunk(id=uuid4(), document_id=doc1_id, chunk_index=0,
                              chunk_content="Investment review shows strong performance."),
            score=0.95
        ),
        ChunkSearchResult(
            chunk=DocumentChunk(id=uuid4(), document_id=doc1_id, chunk_index=1,
                              chunk_content="Annual returns exceeded expectations."),
            score=0.82
        ),
        ChunkSearchResult(
            chunk=DocumentChunk(id=uuid4(), document_id=doc1_id, chunk_index=2,
                              chunk_content="Portfolio rebalancing recommended."),
            score=0.75
        ),
        # Doc2 - 1 chunk, score 0.88
        ChunkSearchResult(
            chunk=DocumentChunk(id=uuid4(), document_id=doc2_id, chunk_index=0,
                              chunk_content="Market outlook remains positive for equities."),
            score=0.88
        ),
        # Doc3 - 2 chunks, highest score is 0.80
        ChunkSearchResult(
            chunk=DocumentChunk(id=uuid4(), document_id=doc3_id, chunk_index=0,
                              chunk_content="Client portfolio summary for review."),
            score=0.80
        ),
        ChunkSearchResult(
            chunk=DocumentChunk(id=uuid4(), document_id=doc3_id, chunk_index=1,
                              chunk_content="Asset allocation breakdown by category."),
            score=0.70
        ),
    ]

    mock_chunk_search_service.search.return_value = chunk_results
    # Batch fetch all 3 documents
    mock_document_repository.get_by_ids.return_value = [doc1, doc2, doc3]

    # Act
    results = await document_search_service.search("investment portfolio", top_k=10)

    # Assert - Should return 3 documents, ordered by highest chunk score
    assert len(results) == 3
    assert results[0].document.id == doc1_id
    assert results[0].score == 0.95
    assert results[1].document.id == doc2_id
    assert results[1].score == 0.88
    assert results[2].document.id == doc3_id
    assert results[2].score == 0.80


@pytest.mark.asyncio
async def test_search_no_results(document_search_service, mock_chunk_search_service, mock_document_repository):
    """Test search with no matching chunks returns empty list."""
    # Arrange
    mock_chunk_search_service.search.return_value = []

    # Act
    results = await document_search_service.search("quantum physics research", top_k=10)

    # Assert
    assert len(results) == 0


@pytest.mark.asyncio
async def test_search_empty_query_raises_error(document_search_service):
    """Test that empty query raises ValueError."""
    with pytest.raises(ValueError, match="Search query cannot be empty"):
        await document_search_service.search("", top_k=10)

    with pytest.raises(ValueError, match="Search query cannot be empty"):
        await document_search_service.search("   ", top_k=10)


@pytest.mark.asyncio
async def test_search_invalid_top_k_raises_error(document_search_service):
    """Test that invalid top_k raises ValueError."""
    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        await document_search_service.search("test query", top_k=0)

    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        await document_search_service.search("test query", top_k=-5)


@pytest.mark.asyncio
async def test_search_invalid_threshold_raises_error(document_search_service):
    """Test that invalid threshold raises ValueError."""
    with pytest.raises(ValueError, match="threshold must be between -1.0 and 1.0"):
        await document_search_service.search("test query", threshold=-1.5)

    with pytest.raises(ValueError, match="threshold must be between -1.0 and 1.0"):
        await document_search_service.search("test query", threshold=1.5)


@pytest.mark.asyncio
async def test_search_with_custom_threshold(document_search_service, mock_chunk_search_service, mock_document_repository):
    """Test search with custom similarity threshold."""
    # Arrange
    doc_id = uuid4()
    document = Document(
        id=doc_id,
        client_id=uuid4(),
        title="Regulatory Compliance Report",
        s3_key="compliance/report.pdf"
    )

    chunk_results = [
        ChunkSearchResult(
            chunk=DocumentChunk(id=uuid4(), document_id=doc_id, chunk_index=0,
                              chunk_content="Regulatory compliance requirements overview."),
            score=0.85
        ),
    ]

    mock_chunk_search_service.search.return_value = chunk_results
    mock_document_repository.get_by_ids.return_value = [document]

    # Act
    results = await document_search_service.search("compliance requirements", top_k=5, threshold=0.7)

    # Assert
    assert len(results) == 1
    # Verify custom threshold was passed to chunk search
    mock_chunk_search_service.search.assert_called_once_with(
        query="compliance requirements",
        top_k=25,  # top_k * 5
        similarity_threshold=0.7
    )


@pytest.mark.asyncio
async def test_search_document_not_found_in_repository(document_search_service, mock_chunk_search_service, mock_document_repository):
    """Test handling when document referenced by chunk is not found."""
    # Arrange
    doc_id = uuid4()

    chunk_results = [
        ChunkSearchResult(
            chunk=DocumentChunk(id=uuid4(), document_id=doc_id, chunk_index=0,
                              chunk_content="Test content"),
            score=0.85
        ),
    ]

    mock_chunk_search_service.search.return_value = chunk_results
    mock_document_repository.get_by_ids.return_value = []  # Document not found

    # Act
    results = await document_search_service.search("test query", top_k=10)

    # Assert - Should handle gracefully and return empty results
    assert len(results) == 0


@pytest.mark.asyncio
async def test_search_wealth_management_scenario(document_search_service, mock_chunk_search_service, mock_document_repository):
    """Test realistic wealth management search scenario."""
    # Arrange - Documents related to a high-net-worth client
    estate_doc_id = uuid4()
    tax_doc_id = uuid4()
    investment_doc_id = uuid4()

    estate_doc = Document(
        id=estate_doc_id,
        client_id=uuid4(),
        title="Estate Planning Strategy - Johnson Family",
        s3_key="estate/johnson-family.pdf"
    )
    tax_doc = Document(
        id=tax_doc_id,
        client_id=uuid4(),
        title="Tax Optimization Report 2024",
        s3_key="tax/optimization-2024.pdf"
    )
    investment_doc = Document(
        id=investment_doc_id,
        client_id=uuid4(),
        title="Alternative Investments Portfolio",
        s3_key="investments/alternatives.pdf"
    )

    # Search for "estate planning" should rank estate doc highest
    chunk_results = [
        ChunkSearchResult(
            chunk=DocumentChunk(id=uuid4(), document_id=estate_doc_id, chunk_index=0,
                              chunk_content="Estate planning for intergenerational wealth transfer strategies."),
            score=0.94
        ),
        ChunkSearchResult(
            chunk=DocumentChunk(id=uuid4(), document_id=estate_doc_id, chunk_index=1,
                              chunk_content="Trust structures and estate tax minimization approaches."),
            score=0.89
        ),
        ChunkSearchResult(
            chunk=DocumentChunk(id=uuid4(), document_id=tax_doc_id, chunk_index=0,
                              chunk_content="Estate tax considerations for high-net-worth families."),
            score=0.76
        ),
        ChunkSearchResult(
            chunk=DocumentChunk(id=uuid4(), document_id=investment_doc_id, chunk_index=0,
                              chunk_content="Alternative investments complement traditional estate holdings."),
            score=0.62
        ),
    ]

    mock_chunk_search_service.search.return_value = chunk_results
    # Batch fetch all 3 documents
    mock_document_repository.get_by_ids.return_value = [estate_doc, tax_doc, investment_doc]

    # Act
    results = await document_search_service.search("estate planning strategies", top_k=5)

    # Assert
    assert len(results) == 3
    # Estate planning document should rank first with highest score
    assert results[0].document.title == "Estate Planning Strategy - Johnson Family"
    assert results[0].score == 0.94
    # Tax document should be second
    assert results[1].document.title == "Tax Optimization Report 2024"
    assert results[1].score == 0.76
    # Investment document should be third
    assert results[2].document.title == "Alternative Investments Portfolio"
    assert results[2].score == 0.62
