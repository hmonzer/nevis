"""Tests for unified SearchService."""
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic.v1 import EmailStr

from src.app.core.domain.models import (
    Client,
    ClientSearchResult,
    Document,
    DocumentSearchResult,
    SearchRequest,
)
from src.app.core.services.search_service import SearchService


@pytest.fixture
def mock_client_search_service():
    """Create a mock ClientSearchService."""
    return AsyncMock()


@pytest.fixture
def mock_document_search_service():
    """Create a mock DocumentSearchService."""
    return AsyncMock()


@pytest.fixture
def search_service(mock_client_search_service, mock_document_search_service):
    """Create SearchService with mocked dependencies."""
    return SearchService(
        client_search_service=mock_client_search_service,
        document_search_service=mock_document_search_service,
    )

class TestSearchServiceMixedResults:
    """Test SearchService with mix of clients and documents."""

    @pytest.mark.asyncio
    async def test_search_returns_mixed_results_sorted_by_score(
        self, search_service, mock_client_search_service, mock_document_search_service
    ):
        """Test that search returns mixed results sorted by score descending."""
        # Arrange
        client1 = Client(
            id=uuid4(),
            first_name="Alice",
            last_name="Smith",
            email=EmailStr("alice@example.com"),
            description="Wealth manager",
        )
        client2 = Client(
            id=uuid4(),
            first_name="Bob",
            last_name="Johnson",
            email=EmailStr("bob@example.com"),
            description="Portfolio manager",
        )
        doc1 = Document(
            id=uuid4(),
            client_id=client1.id,
            title="Investment Strategy",
            s3_key="documents/investment-strategy.pdf",
        )
        doc2 = Document(
            id=uuid4(),
            client_id=client1.id,
            title="Market Analysis",
            s3_key="documents/market-analysis.pdf",
        )

        # Mock client search returns 2 results with scores 0.8 and 0.6
        mock_client_search_service.search.return_value = [
            ClientSearchResult(client=client1, score=0.8),
            ClientSearchResult(client=client2, score=0.6),
        ]

        # Mock document search returns 2 results with scores 0.9 and 0.7
        mock_document_search_service.search.return_value = [
            DocumentSearchResult(document=doc1, score=0.9),
            DocumentSearchResult(document=doc2, score=0.7),
        ]

        request = SearchRequest(query="investment portfolio", top_k=10)

        # Act
        results = await search_service.search(request)

        # Assert
        assert len(results) == 4

        # Verify sorting by score descending: 0.9, 0.8, 0.7, 0.6
        assert results[0].type == "DOCUMENT"
        assert results[0].entity.id == doc1.id
        assert results[0].score == 0.9
        assert results[0].rank == 1

        assert results[1].type == "CLIENT"
        assert results[1].entity.id == client1.id
        assert results[1].score == 0.8
        assert results[1].rank == 2

        assert results[2].type == "DOCUMENT"
        assert results[2].entity.id == doc2.id
        assert results[2].score == 0.7
        assert results[2].rank == 3

        assert results[3].type == "CLIENT"
        assert results[3].entity.id == client2.id
        assert results[3].score == 0.6
        assert results[3].rank == 4

    @pytest.mark.asyncio
    async def test_search_respects_top_k_limit(
        self, search_service, mock_client_search_service, mock_document_search_service
    ):
        """Test that search respects top_k limit when combined results exceed it."""
        # Arrange - Create 3 clients and 3 documents (6 total)
        clients = [
            Client(
                id=uuid4(),
                first_name=f"Client{i}",
                last_name="Test",
                email=EmailStr(f"client{i}@test.com"),
                description=f"Client {i}",
            )
            for i in range(3)
        ]
        documents = [
            Document(
                id=uuid4(),
                client_id=uuid4(),
                title=f"Doc {i}",
                s3_key=f"documents/doc-{i}.pdf",
            )
            for i in range(3)
        ]

        # Mix of scores: 0.9, 0.8, 0.7, 0.6, 0.5, 0.4
        mock_client_search_service.search.return_value = [
            ClientSearchResult(client=clients[0], score=0.9),
            ClientSearchResult(client=clients[1], score=0.6),
            ClientSearchResult(client=clients[2], score=0.4),
        ]

        mock_document_search_service.search.return_value = [
            DocumentSearchResult(document=documents[0], score=0.8),
            DocumentSearchResult(document=documents[1], score=0.7),
            DocumentSearchResult(document=documents[2], score=0.5),
        ]

        request = SearchRequest(query="test", top_k=3)

        # Act
        results = await search_service.search(request)

        # Assert - Should only return top 3
        assert len(results) == 3
        assert results[0].score == 0.9
        assert results[0].rank == 1
        assert results[1].score == 0.8
        assert results[1].rank == 2
        assert results[2].score == 0.7
        assert results[2].rank == 3


class TestSearchServiceNoClientsFound:
    """Test SearchService when no clients are found."""

    @pytest.mark.asyncio
    async def test_search_with_no_clients_returns_only_documents(
        self, search_service, mock_client_search_service, mock_document_search_service
    ):
        """Test that search works correctly when no clients are found."""
        # Arrange
        doc1 = Document(
            id=uuid4(),
            client_id=uuid4(),
            title="Report",
            s3_key="documents/financial-report.pdf",
        )
        doc2 = Document(
            id=uuid4(),
            client_id=uuid4(),
            title="Analysis",
            s3_key="documents/market-analysis.pdf",
        )

        mock_client_search_service.search.return_value = []  # No clients found

        mock_document_search_service.search.return_value = [
            DocumentSearchResult(document=doc1, score=0.8),
            DocumentSearchResult(document=doc2, score=0.6),
        ]

        request = SearchRequest(query="financial report", top_k=10)

        # Act
        results = await search_service.search(request)

        # Assert
        assert len(results) == 2
        assert all(result.type == "DOCUMENT" for result in results)
        assert results[0].entity.id == doc1.id
        assert results[0].rank == 1
        assert results[1].entity.id == doc2.id
        assert results[1].rank == 2


class TestSearchServiceNoDocumentsFound:
    """Test SearchService when no documents are found."""

    @pytest.mark.asyncio
    async def test_search_with_no_documents_returns_only_clients(
        self, search_service, mock_client_search_service, mock_document_search_service
    ):
        """Test that search works correctly when no documents are found."""
        # Arrange
        client1 = Client(
            id=uuid4(),
            first_name="Jane",
            last_name="Doe",
            email=EmailStr("jane@example.com"),
            description="Financial advisor",
        )
        client2 = Client(
            id=uuid4(),
            first_name="Mike",
            last_name="Wilson",
            email=EmailStr("mike@example.com"),
            description="Investment consultant",
        )

        mock_client_search_service.search.return_value = [
            ClientSearchResult(client=client1, score=0.9),
            ClientSearchResult(client=client2, score=0.7),
        ]

        mock_document_search_service.search.return_value = []  # No documents found

        request = SearchRequest(query="financial advisor", top_k=10)

        # Act
        results = await search_service.search(request)

        # Assert
        assert len(results) == 2
        assert all(result.type == "CLIENT" for result in results)
        assert results[0].entity.id == client1.id
        assert results[0].rank == 1
        assert results[1].entity.id == client2.id
        assert results[1].rank == 2


class TestSearchServiceEmptyResults:
    """Test SearchService when no results are found at all."""

    @pytest.mark.asyncio
    async def test_search_with_no_results_returns_empty_list(
        self, search_service, mock_client_search_service, mock_document_search_service
    ):
        """Test that search returns empty list when nothing is found."""
        # Arrange
        mock_client_search_service.search.return_value = []
        mock_document_search_service.search.return_value = []

        request = SearchRequest(query="nonexistent query xyz", top_k=10)

        # Act
        results = await search_service.search(request)

        # Assert
        assert len(results) == 0
        assert results == []


class TestSearchServiceExceptionHandling:
    """Test SearchService exception handling from underlying services."""

    @pytest.mark.asyncio
    async def test_search_handles_client_service_exception(
        self, search_service, mock_client_search_service, mock_document_search_service
    ):
        """Test that search continues when client service raises exception."""
        # Arrange
        doc1 = Document(
            id=uuid4(),
            client_id=uuid4(),
            title="Report",
            s3_key="documents/test-report.pdf",
        )

        # Client search raises an exception
        mock_client_search_service.search.side_effect = Exception("Client search failed")

        mock_document_search_service.search.return_value = [
            DocumentSearchResult(document=doc1, score=0.8)
        ]

        request = SearchRequest(query="test", top_k=10)

        # Act
        results = await search_service.search(request)

        # Assert - Should still return document results
        assert len(results) == 1
        assert results[0].type == "DOCUMENT"
        assert results[0].entity.id == doc1.id

    @pytest.mark.asyncio
    async def test_search_handles_document_service_exception(
        self, search_service, mock_client_search_service, mock_document_search_service
    ):
        """Test that search continues when document service raises exception."""
        # Arrange
        client1 = Client(
            id=uuid4(),
            first_name="Test",
            last_name="User",
            email=EmailStr("test@example.com"),
            description="Test client",
        )

        mock_client_search_service.search.return_value = [
            ClientSearchResult(client=client1, score=0.9)
        ]

        # Document search raises exception
        mock_document_search_service.search.side_effect = Exception("Document search failed")

        request = SearchRequest(query="test", top_k=10)

        # Act
        results = await search_service.search(request)

        # Assert - Should still return client results
        assert len(results) == 1
        assert results[0].type == "CLIENT"
        assert results[0].entity.id == client1.id

    @pytest.mark.asyncio
    async def test_search_handles_both_services_exception(
        self, search_service, mock_client_search_service, mock_document_search_service
    ):
        """Test that search returns empty when both services raise exceptions."""
        # Arrange
        mock_client_search_service.search.side_effect = Exception("Client search failed")
        mock_document_search_service.search.side_effect = Exception("Document search failed")

        request = SearchRequest(query="test", top_k=10)

        # Act
        results = await search_service.search(request)

        # Assert - Should return empty list
        assert len(results) == 0


class TestSearchServiceRanking:
    """Test SearchService ranking behavior."""

    @pytest.mark.asyncio
    async def test_search_assigns_correct_ranks(
        self, search_service, mock_client_search_service, mock_document_search_service
    ):
        """Test that ranks are assigned correctly (1-based, sequential)."""
        # Arrange
        clients = [
            Client(
                id=uuid4(),
                first_name=f"Client{i}",
                last_name="Test",
                email=EmailStr(f"client{i}@test.com"),
                description=f"Client {i}",
            )
            for i in range(5)
        ]

        mock_client_search_service.search.return_value = [
            ClientSearchResult(client=clients[0], score=0.95),
            ClientSearchResult(client=clients[1], score=0.85),
            ClientSearchResult(client=clients[2], score=0.75),
            ClientSearchResult(client=clients[3], score=0.65),
            ClientSearchResult(client=clients[4], score=0.55),
        ]

        mock_document_search_service.search.return_value = []

        request = SearchRequest(query="test", top_k=5)

        # Act
        results = await search_service.search(request)

        # Assert
        assert len(results) == 5
        for i, result in enumerate(results, start=1):
            assert result.rank == i

    @pytest.mark.asyncio
    async def test_search_ranks_equal_scores_consistently(
        self, search_service, mock_client_search_service, mock_document_search_service
    ):
        """Test that equal scores get sequential ranks (stable sort)."""
        # Arrange
        client1 = Client(
            id=uuid4(),
            first_name="Alice",
            last_name="Test",
            email=EmailStr("alice@test.com"),
            description="Client 1",
        )
        doc1 = Document(
            id=uuid4(),
            client_id=uuid4(),
            title="Doc 1",
            s3_key="documents/doc-1.pdf",
        )

        # Both have the same score
        mock_client_search_service.search.return_value = [
            ClientSearchResult(client=client1, score=0.8)
        ]

        mock_document_search_service.search.return_value = [
            DocumentSearchResult(document=doc1, score=0.8)
        ]

        request = SearchRequest(query="test", top_k=10)

        # Act
        results = await search_service.search(request)

        # Assert
        assert len(results) == 2
        assert results[0].rank == 1
        assert results[1].rank == 2
        # Both should have score 0.8
        assert results[0].score == 0.8
        assert results[1].score == 0.8


class TestSearchServiceSearchRequestParameters:
    """Test that SearchService correctly passes SearchRequest to underlying services."""

    @pytest.mark.asyncio
    async def test_search_passes_request_to_both_services(
        self, search_service, mock_client_search_service, mock_document_search_service
    ):
        """Test that the same SearchRequest is passed to both services."""
        # Arrange
        mock_client_search_service.search.return_value = []
        mock_document_search_service.search.return_value = []

        request = SearchRequest(query="test query", top_k=5)

        # Act
        await search_service.search(request)

        # Assert - Verify both services were called with the request
        mock_client_search_service.search.assert_called_once_with(request)
        mock_document_search_service.search.assert_called_once_with(request)
