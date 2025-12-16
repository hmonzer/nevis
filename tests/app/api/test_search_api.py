"""Tests for the search API endpoint."""
import pytest
from httpx import AsyncClient, ASGITransport

from src.client import CreateClientRequest
from src.client.schemas import SearchResultTypeEnum

@pytest.mark.asyncio
async def test_search_empty_database(nevis_client):
    """Test search returns empty list when no data exists."""
    results = await nevis_client.search("test query")
    assert results == []


@pytest.mark.asyncio
async def test_search_finds_client_by_description(nevis_client):
    """Test search finds clients by their description."""
    # Create a client with specific description
    request = CreateClientRequest(
        first_name="John",
        last_name="Smith",
        email="john.smith@test.com",
        description="Wealthy investor interested in technology stocks and IPOs"
    )
    created_client = await nevis_client.create_client(request)

    # Search for the client
    results = await nevis_client.search("technology stocks investor")

    # Should find the client
    assert len(results) > 0
    client_results = [r for r in results if r.type == SearchResultTypeEnum.CLIENT]
    assert len(client_results) > 0
    assert any(r.entity.id == created_client.id for r in client_results)


@pytest.mark.asyncio
async def test_search_respects_top_k(nevis_client):
    """Test search respects the top_k limit."""
    # Create multiple clients
    for i in range(5):
        request = CreateClientRequest(
            first_name=f"Client{i}",
            last_name="TestTopK",
            email=f"client{i}@topk.com",
            description="Generic description for testing top_k limit"
        )
        await nevis_client.create_client(request)

    # Search with top_k=3
    results = await nevis_client.search("Generic description", top_k=3)

    # Should return at most 3 results
    assert len(results) <= 3


@pytest.mark.asyncio
async def test_search_results_have_correct_structure(nevis_client):
    """Test search results have correct structure."""
    # Create a client
    request = CreateClientRequest(
        first_name="Alice",
        last_name="Johnson",
        email="alice.johnson@test.com",
        description="Estate planning and wealth transfer specialist"
    )
    await nevis_client.create_client(request)

    # Search
    results = await nevis_client.search("estate planning")

    # Verify structure
    for result in results:
        assert result.type in [SearchResultTypeEnum.CLIENT, SearchResultTypeEnum.DOCUMENT]
        assert result.score is not None
        assert result.rank >= 1
        assert result.entity is not None


@pytest.mark.asyncio
async def test_search_ranks_results_by_relevance(nevis_client):
    """Test search results are ranked by relevance score."""
    # Create clients with varying relevance
    clients_data = [
        ("Tax", "Expert", "tax.expert@test.com", "Tax planning and optimization expert"),
        ("Finance", "Manager", "finance@test.com", "Financial planning services"),
        ("Health", "Care", "health@test.com", "Healthcare administration"),
    ]

    for first, last, email, desc in clients_data:
        request = CreateClientRequest(
            first_name=first,
            last_name=last,
            email=email,
            description=desc
        )
        await nevis_client.create_client(request)

    # Search for tax planning
    results = await nevis_client.search("tax planning")

    # Results should be ordered by score (descending)
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score, "Results should be sorted by score descending"

    # Ranks should be sequential
    for i, result in enumerate(results):
        assert result.rank == i + 1, f"Rank should be {i + 1}, got {result.rank}"


@pytest.mark.asyncio
async def test_search_query_validation(test_app):
    """Test search validates query parameter."""
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Empty query should fail validation
        response = await client.get("/api/v1/search/", params={"q": ""})
        assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_search_top_k_validation(test_app):
    """Test search validates top_k parameter."""
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # top_k = 0 should fail validation
        response = await client.get("/api/v1/search/", params={"q": "test", "top_k": 0})
        assert response.status_code == 422

        # top_k > 100 should fail validation
        response = await client.get("/api/v1/search/", params={"q": "test", "top_k": 101})
        assert response.status_code == 422


@pytest.mark.asyncio
async def test_search_threshold_validation(test_app):
    """Test search validates threshold parameter."""
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # threshold < -1.0 should fail validation
        response = await client.get("/api/v1/search/", params={"q": "test", "threshold": -1.5})
        assert response.status_code == 422

        # threshold > 1.0 should fail validation
        response = await client.get("/api/v1/search/", params={"q": "test", "threshold": 1.5})
        assert response.status_code == 422
