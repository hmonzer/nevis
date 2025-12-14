from uuid import uuid4

import pytest

from src.shared.database.unit_of_work import UnitOfWork
from tests.shared.database.mock_entities import UserModel, UserMapper, get_test_entity_mapper
from tests.shared.database.mock_repository import UserRepository


@pytest.fixture
def user_repository(clean_database):
    """Create a user repository."""
    return UserRepository(clean_database, UserMapper())


@pytest.fixture
def unit_of_work(clean_database):
    """Create a unit of work."""
    return UnitOfWork(clean_database,  get_test_entity_mapper())


@pytest.mark.asyncio
async def test_persist_user_via_unit_of_work(unit_of_work, user_repository):
    """Test persisting a user entity via unit of work."""
    # Arrange
    user_id = uuid4()
    user_model = UserModel(
        id=user_id,
        name="John Doe",
        email="john.doe@example.com"
    )

    # Act - Persist the user
    async with unit_of_work:
        unit_of_work.add(user_model)

    # Assert - Retrieve and verify
    retrieved_user = await user_repository.get_by_id(user_id)
    assert retrieved_user is not None
    assert retrieved_user.id == user_id
    assert retrieved_user.name == "John Doe"
    assert retrieved_user.email == "john.doe@example.com"


@pytest.mark.asyncio
async def test_retrieve_user_via_repository(unit_of_work, user_repository):
    """Test retrieving a user entity via repository."""
    # Arrange - Create multiple users
    user1_id = uuid4()
    user1 = UserModel(id=user1_id, name="Alice", email="alice@example.com")

    user2_id = uuid4()
    user2 = UserModel(id=user2_id, name="Bob", email="bob@example.com")

    async with unit_of_work:
        unit_of_work.add(user1)
        unit_of_work.add(user2)

    # Act & Assert - Get by ID
    retrieved_user1 = await user_repository.get_by_id(user1_id)
    assert retrieved_user1 is not None
    assert retrieved_user1.name == "Alice"

    # Act & Assert - Get by email
    retrieved_user2 = await user_repository.get_by_email("bob@example.com")
    assert retrieved_user2 is not None
    assert retrieved_user2.id == user2_id
    assert retrieved_user2.name == "Bob"

    # Act & Assert - Get all
    all_users = await user_repository.get_all()
    assert len(all_users) == 2
    assert {user.name for user in all_users} == {"Alice", "Bob"}


@pytest.mark.asyncio
async def test_update_user_via_unit_of_work(unit_of_work, user_repository):
    """Test updating a user entity via unit of work."""
    # Arrange - Create a user
    user_id = uuid4()
    user_model = UserModel(id=user_id, name="Charlie", email="charlie@example.com")

    async with unit_of_work:
        unit_of_work.add(user_model)

    # Act - Update the user
    updated_user = UserModel(id=user_id, name="Charlie Brown", email="charlie.brown@example.com")
    async with unit_of_work:
        await unit_of_work.update(updated_user)

    # Assert
    retrieved_user = await user_repository.get_by_id(user_id)
    assert retrieved_user is not None
    assert retrieved_user.name == "Charlie Brown"
    assert retrieved_user.email == "charlie.brown@example.com"


@pytest.mark.asyncio
async def test_rollback_on_error(unit_of_work, user_repository):
    """Test that changes are rolled back when an error occurs."""
    # Arrange
    user_id = uuid4()
    user_model = UserModel(id=user_id, name="Dave", email="dave@example.com")

    # Act - Try to persist but raise an error
    try:
        async with unit_of_work:
            unit_of_work.add(user_model)
            raise ValueError("Simulated error")
    except ValueError:
        pass

    # Assert - User should not be persisted
    retrieved_user = await user_repository.get_by_id(user_id)
    assert retrieved_user is None


@pytest.mark.asyncio
async def test_get_nonexistent_user(user_repository):
    """Test retrieving a user that doesn't exist."""
    # Act
    result = await user_repository.get_by_id(uuid4())

    # Assert
    assert result is None
