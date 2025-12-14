# Nevis API - Coding Guidelines

## Architecture Overview

This project follows a **Clean Architecture** pattern with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│ API Layer (FastAPI)                                         │
│ - Routes (src/app/api/v1/*.py)                             │
│ - API Schemas (src/client/schemas.py)                      │
│ - API Mappers (src/app/api/mappers.py)                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Domain Layer                                                │
│ - Domain Models (src/app/core/domain/models.py)            │
│ - Services (src/app/core/services/*.py)                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Infrastructure Layer                                        │
│ - DB Entities (src/app/infrastructure/entities/*.py)       │
│ - Repositories (src/app/infrastructure/*_repository.py)    │
│ - Infra Mappers (src/app/infrastructure/mappers/*.py)      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Shared Layer                                                │
│ - Database, UnitOfWork, Base Classes (src/shared/*)        │
└─────────────────────────────────────────────────────────────┘
```

## Adding a New API Endpoint

Follow these steps in order to add a new endpoint for a domain entity:

### 1. Define the Database Entity

**Location:** `src/app/infrastructure/entities/{entity_name}_entity.py`

```python
from datetime import datetime
from uuid import UUID, uuid4
from sqlalchemy import String, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column
from src.shared.database.database import Base

class EntityNameEntity(Base):
    """SQLAlchemy model for EntityName table."""
    __tablename__ = "entity_names"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    field_name: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
```

**Guidelines:**
- Use `Mapped[Type]` type hints for all columns
- Always include `id: UUID` and `created_at: datetime`
- Set appropriate constraints (unique, index, nullable)
- Use singular form for table name in plural (e.g., "clients" for Client)

### 2. Define the Domain Model

**Location:** `src/app/core/domain/models.py`

```python
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field

class EntityName(BaseModel):
    """Domain model for EntityName used in business logic."""
    id: UUID
    field_name: str = Field(..., min_length=1, description="Field description")
    created_at: datetime

    model_config = {"from_attributes": True}
```

**Guidelines:**
- Use Pydantic for validation
- Include `model_config = {"from_attributes": True}` for ORM compatibility
- Add field validators for business rules
- Keep validation consistent with API schemas

### 3. Create the Infrastructure Mapper

**Location:** `src/app/infrastructure/mappers/{entity_name}_mapper.py`

```python
from src.shared.database.base_mapper import BaseEntityMapper
from src.app.core.domain.models import EntityName
from src.app.infrastructure.entities.{entity_name}_entity import EntityNameEntity

class EntityNameMapper(BaseEntityMapper[EntityName, EntityNameEntity]):
    """Mapper for converting between EntityName domain model and EntityNameEntity."""

    @staticmethod
    def to_entity(model_instance: EntityName) -> EntityNameEntity:
        """Convert domain model to database entity."""
        return EntityNameEntity(
            id=model_instance.id,
            field_name=model_instance.field_name,
            created_at=model_instance.created_at,
        )

    @staticmethod
    def to_model(entity: EntityNameEntity) -> EntityName:
        """Convert database entity to domain model."""
        return EntityName(
            id=entity.id,
            field_name=entity.field_name,
            created_at=entity.created_at,
        )
```

**Guidelines:**
- Inherit from `BaseEntityMapper[DomainModel, Entity]`
- Both methods must be `@staticmethod`
- Handle type conversions (e.g., `EmailStr` ↔ `str`)

### 4. Create the Repository

**Location:** `src/app/infrastructure/{entity_name}_repository.py`

```python
from uuid import UUID
from typing import Optional
from sqlalchemy import select

from src.app.core.domain.models import EntityName
from src.shared.database.base_repo import BaseRepository
from src.shared.database.database import Database
from src.app.infrastructure.entities.{entity_name}_entity import EntityNameEntity
from src.app.infrastructure.mappers.{entity_name}_mapper import EntityNameMapper

class EntityNameRepository(BaseRepository[EntityNameEntity, EntityName]):
    """Repository for EntityName operations."""

    def __init__(self, db: Database, mapper: EntityNameMapper):
        super().__init__(db, mapper)

    async def get_by_id(self, entity_id: UUID) -> Optional[EntityName]:
        """Get an entity by ID."""
        return await self.find_one(
            select(EntityNameEntity).where(EntityNameEntity.id == entity_id)
        )

    async def get_by_field(self, field_value: str) -> Optional[EntityName]:
        """Get an entity by custom field."""
        return await self.find_one(
            select(EntityNameEntity).where(EntityNameEntity.field_name == field_value)
        )
```

**Guidelines:**
- Inherit from `BaseRepository[Entity, DomainModel]`
- Use `find_one()` for single results (returns `Optional[Model]`)
- Use `find_all()` for multiple results (returns `list[Model]`)
- Build queries using SQLAlchemy `select()` statements

### 5. Create the Service

**Location:** `src/app/core/services/{entity_name}_service.py`

```python
from uuid import UUID, uuid4
from datetime import datetime, UTC
from sqlalchemy.exc import IntegrityError

from src.app.core.domain.models import EntityName
from src.shared.database.unit_of_work import UnitOfWork
from src.client.schemas import CreateEntityNameRequest
from src.app.infrastructure.{entity_name}_repository import EntityNameRepository

class EntityNameService:
    """Service for handling EntityName business logic."""

    def __init__(self, repository: EntityNameRepository, unit_of_work: UnitOfWork):
        self.repository = repository
        self.unit_of_work = unit_of_work

    async def create_entity(self, request: CreateEntityNameRequest) -> EntityName:
        """Create a new entity."""
        # Create domain model with generated ID and timestamp
        entity = EntityName(
            id=uuid4(),
            field_name=request.field_name,
            created_at=datetime.now(UTC),
        )

        # Persist using unit of work
        try:
            async with self.unit_of_work:
                self.unit_of_work.add(entity)
        except IntegrityError as e:
            raise ValueError(f"Constraint violation: {str(e)}") from e

        return entity

    async def get_entity(self, entity_id: UUID) -> EntityName:
        """Get an entity by ID."""
        entity = await self.repository.get_by_id(entity_id)
        if not entity:
            raise ValueError(f"EntityName with ID {entity_id} not found")
        return entity
```

**Guidelines:**
- **Write operations**: Use `UnitOfWork` with context manager
- **Read operations**: Use repository methods directly
- Generate `id` with `uuid4()` and `created_at` with `datetime.now(UTC)`
- Raise `ValueError` for business rule violations
- Catch `IntegrityError` for DB constraint violations

### 6. Define API Schemas

**Location:** `src/client/schemas.py`

```python
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field, field_validator

class CreateEntityNameRequest(BaseModel):
    """Request schema for creating a new entity."""
    field_name: str = Field(..., min_length=1, description="Field description")

    @field_validator("field_name")
    @classmethod
    def validate_field(cls, v: str) -> str:
        """Custom validation logic."""
        if not v or not v.strip():
            raise ValueError("Field cannot be blank")
        return v.strip()

class EntityNameResponse(BaseModel):
    """Response schema for entity data returned by the API."""
    id: UUID
    field_name: str
    created_at: datetime

    model_config = {"from_attributes": True}
```

**Guidelines:**
- Separate request and response schemas
- Request schemas: validation and transformation
- Response schemas: include `model_config = {"from_attributes": True}`
- Use `field_validator` for custom validation

### 7. Create API Mapper

**Location:** `src/app/api/mappers.py`

```python
from src.app.core.domain.models import EntityName
from src.client.schemas import EntityNameResponse

def to_entity_name_response(entity: EntityName) -> EntityNameResponse:
    """
    Convert EntityName domain model to EntityNameResponse API schema.

    Args:
        entity: Domain model

    Returns:
        API response schema
    """
    return EntityNameResponse(
        id=entity.id,
        field_name=entity.field_name,
        created_at=entity.created_at,
    )
```

**Guidelines:**
- One function per mapping direction
- Use descriptive names: `to_{schema_name}()`
- Include docstring with Args and Returns

### 8. Setup Dependencies

**Location:** `src/app/api/dependencies.py`

```python
from typing import AsyncGenerator
from fastapi import Depends

from src.app.core.services.{entity_name}_service import EntityNameService
from src.app.infrastructure.{entity_name}_repository import EntityNameRepository
from src.app.infrastructure.mappers.{entity_name}_mapper import EntityNameMapper
from src.shared.database.entity_mapper import EntityMapper
from src.app.core.domain.models import EntityName

def get_entity_name_mapper() -> EntityNameMapper:
    """Get EntityName mapper instance."""
    return EntityNameMapper()

def get_entity_name_repository(
    db: Database = Depends(get_database),
    mapper: EntityNameMapper = Depends(get_entity_name_mapper)
) -> EntityNameRepository:
    """Get EntityName repository instance."""
    return EntityNameRepository(db, mapper)

def get_entity_mapper(
    entity_name_mapper: EntityNameMapper = Depends(get_entity_name_mapper),
    # ... other mappers
) -> EntityMapper:
    """Get entity mapper with all model mappings."""
    return EntityMapper(
        entity_mappings={
            EntityName: entity_name_mapper.to_entity,
            # ... other mappings
        }
    )

def get_entity_name_service(
    repository: EntityNameRepository = Depends(get_entity_name_repository),
    unit_of_work: UnitOfWork = Depends(get_unit_of_work)
) -> EntityNameService:
    """Get EntityName service instance."""
    return EntityNameService(repository, unit_of_work)
```

**Guidelines:**
- Create dependency functions for each layer: mapper → repository → service
- Add mapper to `get_entity_mapper()` for UnitOfWork support
- Use FastAPI's `Depends()` for dependency injection

### 9. Create API Endpoints

**Location:** `src/app/api/v1/{entity_name}s.py`

```python
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status

from src.app.core.services.{entity_name}_service import EntityNameService
from src.client.schemas import CreateEntityNameRequest, EntityNameResponse
from src.app.api.dependencies import get_entity_name_service
from src.app.api.mappers import to_entity_name_response

router = APIRouter(prefix="/{entity_name}s", tags=["{entity_name}s"])

@router.post("/", response_model=EntityNameResponse, status_code=status.HTTP_201_CREATED)
async def create_entity(
    request: CreateEntityNameRequest,
    service: EntityNameService = Depends(get_entity_name_service)
) -> EntityNameResponse:
    """Create a new entity."""
    try:
        entity = await service.create_entity(request)
        return to_entity_name_response(entity)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@router.get("/{entity_id}", response_model=EntityNameResponse)
async def get_entity(
    entity_id: UUID,
    service: EntityNameService = Depends(get_entity_name_service)
) -> EntityNameResponse:
    """Get an entity by ID."""
    try:
        entity = await service.get_entity(entity_id)
        return to_entity_name_response(entity)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
```

**Guidelines:**
- Use plural for endpoint prefix (e.g., `/clients`)
- Inject service via `Depends(get_entity_name_service)`
- Convert `ValueError` → `HTTPException` with appropriate status codes
- Use mapper to convert domain model → API response
- Specify `response_model` and `status_code` explicitly

### 10. Register Router

**Location:** `src/app/main.py`

```python
from src.app.api.v1 import {entity_name}s

def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    # ... existing code ...

    # Include routers
    app.include_router({entity_name}s.router, prefix="/api/v1")

    return app
```

## Testing Guidelines

### Test Structure

**Location:** `tests/app/test_{entity_name}_api.py`

```python
from typing import cast
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport, HTTPStatusError
from pydantic import ValidationError

from src.app.main import create_app
from src.client import NevisClient, CreateEntityNameRequest

@pytest_asyncio.fixture(scope="module")
async def test_app(test_settings_override):
    """
    Create test application with test database.
    Module-scoped for better performance.
    """
    app = create_app()
    yield app

@pytest_asyncio.fixture
async def nevis_client(test_app, clean_database):
    """Create Nevis client for testing with clean database."""
    transport = ASGITransport(app=test_app)
    http_client = AsyncClient(transport=transport, base_url="http://test")
    client = NevisClient(base_url="http://test", client=http_client)

    async with client:
        yield client

@pytest.mark.asyncio
async def test_create_entity(nevis_client):
    """Test creating an entity via API."""
    request = CreateEntityNameRequest(field_name="test")
    response = await nevis_client.create_entity(request)

    assert response.field_name == "test"
    assert response.id is not None

@pytest.mark.asyncio
async def test_validation_error():
    """Test validation errors."""
    with pytest.raises(ValidationError) as exc_info:
        CreateEntityNameRequest(field_name="")

    error = cast(ValidationError, exc_info.value)
    errors = error.errors()
    assert any(error["loc"] == ("field_name",) for error in errors)
```

**Guidelines:**
- **Fixture scoping**: `test_app` is module-scoped, `nevis_client` depends on `clean_database` for isolation
- **Settings override**: Use `test_settings_override` fixture (centralized in `conftest.py`)
- **Type safety**: Use `cast()` for exception type narrowing
- **API tests**: Test via client, not direct service calls
- **Test isolation**: Each test gets a clean database via `clean_database` dependency

### Test Configuration

**Location:** `tests/conftest.py`

- `postgres_container`: Module-scoped PostgreSQL container
- `async_db_url`: Module-scoped async database URL
- `test_settings_override`: Module-scoped settings configuration (DATABASE_URL override)
- `db`: Function-scoped database instance
- `clean_database`: Function-scoped fixture that drops/creates tables

## Key Patterns

### Data Flow

**CREATE Operation:**
```
API Request Schema → Service → Domain Model → UnitOfWork → Mapper → DB Entity → Database
Database → DB Entity → Mapper → Domain Model → API Mapper → API Response Schema
```

**READ Operation:**
```
API Path Param → Service → Repository → SQL Query → DB Entity → Mapper → Domain Model → API Mapper → API Response Schema
```

### Error Handling

1. **API Layer**: Catches `ValueError`, converts to `HTTPException`
2. **Service Layer**: Raises `ValueError` for business rule violations
3. **Database Layer**: Raises `IntegrityError` for constraint violations
4. **Service catches**: `IntegrityError` → raises `ValueError` with context

### Type Safety

- **Generic base classes**: `BaseRepository[TEntity, TModel]`, `BaseEntityMapper[TModel, TEntity]`
- **Type hints**: Use throughout (async, Optional, list, etc.)
- **Pydantic models**: For validation and serialization
- **SQLAlchemy Mapped**: For ORM type safety

### Transaction Management

- **Reads**: Direct repository calls (no transaction needed)
- **Writes**: Always use `async with self.unit_of_work:` context manager
- **Rollback**: Automatic on exception in context manager

## Common Pitfalls to Avoid

1. **Don't bypass the service layer** - API should always call services, not repositories
2. **Don't use domain models in API directly** - Always use API schemas and mappers
3. **Don't create entities in repositories** - Repositories only query, services create domain models
4. **Don't forget to register mappers** - Add to `get_entity_mapper()` in dependencies
5. **Don't skip validation** - Add validators to both API schemas and domain models
6. **Don't commit directly** - Always use UnitOfWork context manager for writes
7. **Don't use function-scoped fixtures unnecessarily** - Use module scope for expensive resources like test_app

## Code Quality

- **Type checking**: Run `uvx ty check` before committing
- **Testing**: All tests must pass via `uv run pytest`
- **Import organization**: Standard library → Third party → Local imports
- **Async/await**: All I/O operations must be async
- **Docstrings**: Include for all public classes and methods
