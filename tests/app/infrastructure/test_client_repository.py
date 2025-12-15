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
