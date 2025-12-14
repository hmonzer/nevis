from uuid import uuid4
from datetime import datetime, UTC

import pytest
import pytest_asyncio
from pydantic.v1 import EmailStr

from src.app.core.domain.models import Client
from src.shared.database.unit_of_work import UnitOfWork
from src.shared.database.entity_mapper import EntityMapper

from src.app.infrastructure.client_repository import ClientRepository
from src.app.infrastructure.mappers.client_mapper import ClientMapper


@pytest_asyncio.fixture
async def client_repository(clean_database):
    """Create a client repository."""
    return ClientRepository(clean_database, ClientMapper())


@pytest_asyncio.fixture
async def unit_of_work(clean_database):
    """Create a unit of work."""
    entity_mapper = EntityMapper(
        entity_mappings={
            Client: ClientMapper.to_entity,
        }
    )
    return UnitOfWork(clean_database, entity_mapper)


@pytest.mark.asyncio
async def test_get_client_by_id(client_repository, unit_of_work):
    """Test retrieving a client by ID."""
    # Arrange - Create a client
    client_id = uuid4()
    client = Client(
        id=client_id,
        first_name="John",
        last_name="Doe",
        email=EmailStr("john.doe@example.com"),
        description="Test client"
    )

    # Persist the client
    async with unit_of_work:
        unit_of_work.add(client)

    # Act - Retrieve by ID
    retrieved_client = await client_repository.get_by_id(client_id)

    # Assert
    assert retrieved_client is not None
    assert retrieved_client.id == client_id
    assert retrieved_client.first_name == "John"
    assert retrieved_client.last_name == "Doe"
    assert retrieved_client.email == "john.doe@example.com"
    assert retrieved_client.description == "Test client"


@pytest.mark.asyncio
async def test_get_client_by_id_not_found(client_repository):
    """Test retrieving a non-existent client by ID returns None."""
    # Act
    non_existent_id = uuid4()
    result = await client_repository.get_by_id(non_existent_id)

    # Assert
    assert result is None


@pytest.mark.asyncio
async def test_get_client_by_email(client_repository, unit_of_work):
    """Test retrieving a client by email."""
    # Arrange - Create a client
    client = Client(
        first_name="Jane",
        last_name="Smith",
        email=EmailStr("jane.smith@example.com"),
        description="Test client"
    )

    # Persist the client
    async with unit_of_work:
        unit_of_work.add(client)

    # Act - Retrieve by email
    retrieved_client = await client_repository.get_by_email("jane.smith@example.com")

    # Assert
    assert retrieved_client is not None
    assert retrieved_client.id == client.id
    assert retrieved_client.first_name == "Jane"
    assert retrieved_client.last_name == "Smith"
    assert retrieved_client.email == "jane.smith@example.com"


@pytest.mark.asyncio
async def test_get_client_by_email_not_found(client_repository):
    """Test retrieving a non-existent client by email returns None."""
    # Act
    result = await client_repository.get_by_email("nonexistent@example.com")

    # Assert
    assert result is None


@pytest.mark.asyncio
async def test_search_by_email_partial_match(client_repository, unit_of_work):
    """Test searching for clients by partial email match."""
    # Arrange - Create clients
    client1 = Client(
        first_name="John",
        last_name="Doe",
        email=EmailStr("john.doe@neviswealth.com"),
        description="Wealth management client"
    )
    client2 = Client(
        first_name="Jane",
        last_name="Smith",
        email=EmailStr("jane.smith@example.com"),
        description="Regular client"
    )

    async with unit_of_work:
        unit_of_work.add(client1)
        unit_of_work.add(client2)

    # Act - Search for "NevisWealth" which should match "john.doe@neviswealth.com"
    results = await client_repository.search("NevisWealth")

    # Assert
    assert len(results) == 1
    assert results[0].email == "john.doe@neviswealth.com"
    assert results[0].first_name == "John"


@pytest.mark.asyncio
async def test_search_by_first_name(client_repository, unit_of_work):
    """Test searching for clients by first name."""
    # Arrange
    client1 = Client(
        first_name="Alexander",
        last_name="Johnson",
        email=EmailStr("alex.johnson@example.com"),
        description="Client A"
    )
    client2 = Client(
        first_name="Alexandra",
        last_name="Williams",
        email=EmailStr("alexandra.williams@example.com"),
        description="Client B"
    )

    async with unit_of_work:
        unit_of_work.add(client1)
        unit_of_work.add(client2)

    # Act - Search for "Alex" which should match both Alexander and Alexandra
    results = await client_repository.search("Alex")

    # Assert - Both should be found
    assert len(results) == 2
    first_names = {client.first_name for client in results}
    assert "Alexander" in first_names
    assert "Alexandra" in first_names


@pytest.mark.asyncio
async def test_search_by_last_name(client_repository, unit_of_work):
    """Test searching for clients by last name."""
    # Arrange
    client = Client(
        first_name="Michael",
        last_name="O'Brien",
        email=EmailStr("michael.ob@example.com"),
        description="Test client"
    )

    async with unit_of_work:
        unit_of_work.add(client)

    # Act - Search for "OBrien" which should match "O'Brien"
    results = await client_repository.search("OBrien")

    # Assert
    assert len(results) == 1
    assert results[0].last_name == "O'Brien"


@pytest.mark.asyncio
async def test_search_by_description(client_repository, unit_of_work):
    """Test searching for clients by description."""
    # Arrange
    client1 = Client(
        first_name="Sarah",
        last_name="Connor",
        email=EmailStr("sarah.connor@example.com"),
        description="Technology sector executive"
    )
    client2 = Client(
        first_name="John",
        last_name="Smith",
        email=EmailStr("john.smith@example.com"),
        description="Real estate investor"
    )

    async with unit_of_work:
        unit_of_work.add(client1)
        unit_of_work.add(client2)

    # Act - Search for "technology"
    results = await client_repository.search("technology")

    # Assert
    assert len(results) == 1
    assert results[0].first_name == "Sarah"
    assert "Technology" in results[0].description


@pytest.mark.asyncio
async def test_search_with_null_description(client_repository, unit_of_work):
    """Test searching for clients when some have null descriptions."""
    # Arrange
    client1 = Client(
        first_name="Robert",
        last_name="Taylor",
        email=EmailStr("robert.taylor@example.com"),
        description=None,  # Null descriptio
    )
    client2 = Client(
        first_name="Emily",
        last_name="Davis",
        email=EmailStr("emily.davis@example.com"),
        description="Financial advisor"
    )

    async with unit_of_work:
        unit_of_work.add(client1)
        unit_of_work.add(client2)

    # Act - Search for "Robert" (should match first name despite null description)
    results = await client_repository.search("Robert")

    # Assert
    assert len(results) == 1
    assert results[0].first_name == "Robert"
    assert results[0].description is None


@pytest.mark.asyncio
async def test_search_no_matches(client_repository, unit_of_work):
    """Test searching with a query that has no matches."""
    # Arrange
    client = Client(
        first_name="David",
        last_name="Wilson",
        email=EmailStr("david.wilson@example.com"),
        description="Portfolio manager"
    )

    async with unit_of_work:
        unit_of_work.add(client)

    # Act - Search for something completely unrelated
    results = await client_repository.search("xyzabc123")

    # Assert
    assert len(results) == 0


@pytest.mark.asyncio
async def test_search_orders_by_relevance(client_repository, unit_of_work):
    """Test that search results are ordered by relevance (highest similarity first)."""
    # Arrange - Create clients with varying similarity to search term
    client1 = Client(
        first_name="Christopher",
        last_name="Anderson",
        email=EmailStr("christopher.anderson@example.com"),
        description="Investment banker"
    )
    client2 = Client(
        first_name="Chris",
        last_name="Martin",
        email=EmailStr("chris.martin@example.com"),
        description="Trader"
    )
    client3 = Client(
        first_name="Christine",
        last_name="Brown",
        email=EmailStr("christine.brown@example.com"),
        description="Analyst"
    )

    async with unit_of_work:
        unit_of_work.add(client1)
        unit_of_work.add(client2)
        unit_of_work.add(client3)

    # Act - Search for "Chris" - exact match should be first
    results = await client_repository.search("Chris")

    # Assert - "Chris" should be first (exact match), others follow
    assert len(results) >= 1
    assert results[0].first_name == "Chris"


@pytest.mark.asyncio
async def test_search_with_custom_threshold(client_repository, unit_of_work):
    """Test search with custom similarity threshold."""
    # Arrange
    client = Client(
        first_name="William",
        last_name="Taylor",
        email=EmailStr("william.taylor@example.com"),
        description="Consultant"
    )

    client = Client(
        first_name="Wilbur",
        last_name="Taylor",
        email=EmailStr("william.taylor@example.com"),
        description="Consultant"
    )

    async with unit_of_work:
        unit_of_work.add(client)

    # Act - Search with very high threshold (strict matching)
    results_strict = await client_repository.search("Will", threshold=0.5)
    # Search with low threshold (loose matching)
    results_loose = await client_repository.search("Will", threshold=0.1)

    # Assert - Loose threshold should find more or equal results
    assert len(results_loose) > len(results_strict)
