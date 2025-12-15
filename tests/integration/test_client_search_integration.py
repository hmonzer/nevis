"""Integration tests for ClientSearchService."""
from uuid import uuid4

import pytest
import pytest_asyncio
from pydantic.v1 import EmailStr

from src.app.core.domain.models import Client, SearchRequest
from src.app.core.services.client_search_service import ClientSearchService
from src.app.infrastructure.client_search_repository import ClientSearchRepository
from src.app.infrastructure.mappers.client_mapper import ClientMapper
from src.shared.database.unit_of_work import UnitOfWork
from src.shared.database.entity_mapper import EntityMapper


@pytest_asyncio.fixture
async def client_search_repository(clean_database):
    """Create a client search repository."""
    return ClientSearchRepository(clean_database, ClientMapper())


@pytest_asyncio.fixture
async def unit_of_work(clean_database):
    """Create a unit of work."""
    entity_mapper = EntityMapper(
        entity_mappings={
            Client: ClientMapper.to_entity,
        }
    )
    return UnitOfWork(clean_database, entity_mapper)


@pytest_asyncio.fixture
async def client_search_service(client_search_repository):
    """Create a client search service."""
    return ClientSearchService(client_search_repository)


@pytest_asyncio.fixture
async def wealth_management_clients(unit_of_work):
    """Create sample wealth management clients for testing."""
    clients = [
        Client(
            id=uuid4(),
            first_name="Jonathan",
            last_name="Sterling",
            email=EmailStr("jonathan.sterling@goldmansachs.com"),
            description="Senior Investment Banker specializing in M&A transactions for technology sector clients. "
                       "Manages high-net-worth portfolios exceeding $50M."
        ),
        Client(
            id=uuid4(),
            first_name="Elizabeth",
            last_name="Chen",
            email=EmailStr("elizabeth.chen@morganstanley.com"),
            description="Wealth Management Advisor focused on ultra-high-net-worth families. "
                       "Expertise in estate planning and multi-generational wealth transfer strategies."
        ),
        Client(
            id=uuid4(),
            first_name="Robert",
            last_name="Patterson",
            email=EmailStr("robert.patterson@realtypartners.com"),
            description="Commercial Real Estate Developer and investor. "
                       "Portfolio includes office buildings and mixed-use developments across major metropolitan areas."
        ),
        Client(
            id=uuid4(),
            first_name="Sarah",
            last_name="Goldman",
            email=EmailStr("sarah.goldman@hedgecapital.com"),
            description="Portfolio Manager at quantitative hedge fund. "
                       "Focuses on algorithmic trading strategies and risk management for institutional clients."
        ),
        Client(
            id=uuid4(),
            first_name="Michael",
            last_name="Torres",
            email=EmailStr("michael.torres@techventures.com"),
            description="Technology Sector Executive and angel investor. "
                       "Former CEO of multiple successful startups, now advising early-stage technology companies."
        ),
        Client(
            id=uuid4(),
            first_name="Amanda",
            last_name="Richardson",
            email=EmailStr("amanda.richardson@privatebank.com"),
            description="Private Banking Relationship Manager serving corporate executives and entrepreneurs. "
                       "Specializes in credit facilities and liquidity solutions."
        ),
    ]

    async with unit_of_work:
        for client in clients:
            unit_of_work.add(client)

    return clients


@pytest.mark.asyncio
async def test_search_investment_banker(client_search_service, wealth_management_clients):
    """Test searching for investment banking professionals."""
    # Act
    request = SearchRequest(query="investment banker", top_k=5, threshold=0.1)
    results = await client_search_service.search(request)

    # Assert - Jonathan Sterling should rank highest
    assert len(results) > 0
    assert results[0].client.first_name == "Jonathan"
    assert results[0].client.last_name == "Sterling"
    assert "Investment Banker" in results[0].client.description

    # Verify score is valid
    assert 0.0 <= results[0].score <= 1.0

    # Verify results are ordered by score descending
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score


@pytest.mark.asyncio
async def test_search_wealth_management(client_search_service, wealth_management_clients):
    """Test searching for wealth management professionals."""
    # Act
    request = SearchRequest(query="wealth management", top_k=5, threshold=0.1)
    results = await client_search_service.search(request)

    # Assert - Elizabeth Chen should rank highest
    assert len(results) > 0
    assert results[0].client.first_name == "Elizabeth"
    assert results[0].client.last_name == "Chen"
    assert "Wealth Management" in results[0].client.description
    assert 0.0 <= results[0].score <= 1.0


@pytest.mark.asyncio
async def test_search_real_estate(client_search_service, wealth_management_clients):
    """Test searching for real estate professionals."""
    # Act
    request = SearchRequest(query="real estate", top_k=5, threshold=0.1)
    results = await client_search_service.search(request)

    # Assert - Robert Patterson should rank highest
    assert len(results) > 0
    assert results[0].client.first_name == "Robert"
    assert results[0].client.last_name == "Patterson"
    assert "Real Estate" in results[0].client.description
    assert 0.0 <= results[0].score <= 1.0


@pytest.mark.asyncio
async def test_search_technology_sector(client_search_service, wealth_management_clients):
    """Test searching for technology sector professionals."""
    # Act
    request = SearchRequest(query="technology", top_k=5, threshold=0.1)
    results = await client_search_service.search(request)

    # Assert - Michael Torres (tech executive) should rank highest
    assert len(results) > 0
    assert results[0].client.first_name == "Michael"
    assert results[0].client.last_name == "Torres"
    assert "Technology" in results[0].client.description

    # Verify all scores are valid
    for result in results:
        assert 0.0 <= result.score <= 1.0


@pytest.mark.asyncio
async def test_search_portfolio_manager(client_search_service, wealth_management_clients):
    """Test searching for portfolio management professionals."""
    # Act
    request = SearchRequest(query="portfolio manager", top_k=5, threshold=0.1)
    results = await client_search_service.search(request)

    # Assert - Sarah Goldman should rank highest
    assert len(results) > 0
    assert results[0].client.first_name == "Sarah"
    assert results[0].client.last_name == "Goldman"
    assert "Portfolio Manager" in results[0].client.description
    assert 0.0 <= results[0].score <= 1.0


@pytest.mark.asyncio
async def test_search_by_company_domain(client_search_service, wealth_management_clients):
    """Test searching by company email domain."""
    # Act - Search for Morgan Stanley
    request = SearchRequest(query="morganstanley", top_k=5, threshold=0.1)
    results = await client_search_service.search(request)

    # Assert - Elizabeth Chen should be found (morganstanley.com email)
    assert len(results) > 0
    assert results[0].client.first_name == "Elizabeth"
    assert "morganstanley.com" in results[0].client.email
    assert 0.0 <= results[0].score <= 1.0


@pytest.mark.asyncio
async def test_search_by_first_name(client_search_service, wealth_management_clients):
    """Test searching by first name."""
    # Act - Search for "Sarah"
    request = SearchRequest(query="Sarah", top_k=5)
    results = await client_search_service.search(request)

    # Assert - Sarah Goldman should be found
    assert len(results) > 0
    assert results[0].client.first_name == "Sarah"
    assert results[0].client.last_name == "Goldman"
    assert 0.0 <= results[0].score <= 1.0


@pytest.mark.asyncio
async def test_search_with_high_threshold(client_search_service, wealth_management_clients):
    """Test that high threshold returns only close matches."""
    # Act - Search with high threshold (strict matching)
    request_strict = SearchRequest(query="investment banker", top_k=10, threshold=0.4)
    results_strict = await client_search_service.search(request_strict)

    # Search with low threshold (loose matching)
    request_loose = SearchRequest(query="investment banker", top_k=10, threshold=0.1)
    results_loose = await client_search_service.search(request_loose)

    # Assert - Loose threshold should return more or equal results
    assert len(results_loose) >= len(results_strict)

    # All strict results should have high scores
    for result in results_strict:
        assert result.score >= 0.4

    # All loose results should have lower minimum score
    for result in results_loose:
        assert result.score >= 0.1


@pytest.mark.asyncio
async def test_search_with_top_k_limit(client_search_service, wealth_management_clients):
    """Test that top_k parameter limits the number of results."""
    # Act - Search with different top_k values
    request_top_3 = SearchRequest(query="manager", top_k=3)
    results_top_3 = await client_search_service.search(request_top_3)

    request_top_10 = SearchRequest(query="manager", top_k=10)
    results_top_10 = await client_search_service.search(request_top_10)

    # Assert - Should respect top_k limit
    assert len(results_top_3) <= 3
    assert len(results_top_10) <= 10

    # If there are at least 3 results, top_3 should return exactly 3
    if len(results_top_10) >= 3:
        assert len(results_top_3) == 3


@pytest.mark.asyncio
async def test_search_no_matches(client_search_service, wealth_management_clients):
    """Test searching with a query that has no matches."""
    # Act - Search for something completely unrelated
    request = SearchRequest(query="quantum physics researcher", top_k=5)
    results = await client_search_service.search(request)

    # Assert - Should return empty list or very low scores
    assert len(results) == 0 or all(result.score < 0.2 for result in results)


@pytest.mark.asyncio
async def test_search_multi_keyword_query(client_search_service, wealth_management_clients):
    """Test searching with multiple keywords."""
    # Act - Search for "hedge fund portfolio"
    request = SearchRequest(query="hedge fund portfolio", top_k=5, threshold=0.1)
    results = await client_search_service.search(request)

    # Assert - Sarah Goldman (hedge fund portfolio manager) should rank highest
    assert len(results) > 0
    assert results[0].client.first_name == "Sarah"
    assert results[0].client.last_name == "Goldman"
    assert "hedge fund" in results[0].client.description
    assert "Portfolio Manager" in results[0].client.description
    assert 0.0 <= results[0].score <= 1.0
