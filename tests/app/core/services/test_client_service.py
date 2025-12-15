import pytest
import pytest_asyncio
from uuid import uuid4
from src.app.core.services.client_service import ClientService
from src.client.schemas import CreateClientRequest
from src.shared.exceptions import ConflictingEntityFound, EntityNotFound
from src.app.infrastructure.client_repository import ClientRepository
from src.app.infrastructure.mappers.client_mapper import ClientMapper

@pytest_asyncio.fixture
async def client_repository(clean_database):
    """Fixture for ClientRepository with a real database session."""
    return ClientRepository(db=clean_database, mapper=ClientMapper())


@pytest.fixture
def client_service(client_repository, unit_of_work):
    """Fixture for ClientService with real repository and unit of work."""
    return ClientService(repository=client_repository, unit_of_work=unit_of_work)


@pytest.mark.asyncio
async def test_create_client_successfully(client_service, client_repository):
    # Arrange
    request = CreateClientRequest(
        first_name="John",
        last_name="Doe",
        email="john.doe@example.com",
        description="A test client"
    )

    # Act
    created_client = await client_service.create_client(request)

    # Assert
    assert created_client.first_name == request.first_name
    assert created_client.email == request.email
    # Verify it was actually saved to the database
    db_client = await client_repository.get_by_id(created_client.id)
    assert db_client is not None
    assert db_client.email == "john.doe@example.com"


@pytest.mark.asyncio
async def test_create_client_raises_conflict_on_duplicate_email(client_service):
    # Arrange
    request = CreateClientRequest(
        first_name="Jane",
        last_name="Doe",
        email="jane.doe@example.com",
        description="A test client"
    )
    # Create the first client
    await client_service.create_client(request)

    # Act & Assert
    with pytest.raises(ConflictingEntityFound):
        # Try to create another client with the same email
        await client_service.create_client(request)


@pytest.mark.asyncio
async def test_get_client_successfully(client_service, client_repository):
    # Arrange
    request = CreateClientRequest(
        first_name="John",
        last_name="Doe",
        email="john.doe@example.com",
        description="A test client"
    )
    created_client = await client_service.create_client(request)

    # Act
    found_client = await client_service.get_client(created_client.id)

    # Assert
    assert found_client is not None
    assert found_client.id == created_client.id
    assert found_client.email == "john.doe@example.com"


@pytest.mark.asyncio
async def test_get_client_raises_not_found(client_service):
    # Arrange
    non_existent_id = uuid4()

    # Act & Assert
    with pytest.raises(EntityNotFound):
        await client_service.get_client(non_existent_id)


@pytest.mark.asyncio
async def test_get_client_by_email_successfully(client_service):
    # Arrange
    request = CreateClientRequest(
        first_name="John",
        last_name="Doe",
        email="john.doe@example.com",
        description="A test client"
    )
    await client_service.create_client(request)

    # Act
    found_client = await client_service.get_client_by_email("john.doe@example.com")

    # Assert
    assert found_client is not None
    assert found_client.email == "john.doe@example.com"


@pytest.mark.asyncio
async def test_get_client_by_email_raises_not_found(client_service):
    # Arrange
    non_existent_email = "nonexistent@example.com"

    # Act & Assert
    with pytest.raises(EntityNotFound):
        await client_service.get_client_by_email(non_existent_email)
