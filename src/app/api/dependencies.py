from typing import AsyncGenerator
from fastapi import Depends

from src.shared.database.database import Database
from src.shared.database.database_settings import DatabaseSettings
from src.shared.database.unit_of_work import UnitOfWork
from src.shared.database.entity_mapper import EntityMapper
from src.app.domain.models import Client
from src.app.infrastructure.client_repository import ClientRepository
from src.app.infrastructure.mappers.client_mapper import ClientMapper
from src.app.services.client_service import ClientService
from src.app.core.config import get_settings


async def get_database() -> AsyncGenerator[Database, None]:
    """Get database instance."""
    settings = get_settings()
    db_settings = DatabaseSettings(db_url=settings.database_url)
    db = Database(db_settings)
    try:
        yield db
    finally:
        await db._engine.dispose()


def get_client_mapper() -> ClientMapper:
    """Get Client mapper instance."""
    return ClientMapper()


def get_client_repository(
    db: Database = Depends(get_database),
    mapper: ClientMapper = Depends(get_client_mapper)
) -> ClientRepository:
    """Get Client repository instance."""
    return ClientRepository(db, mapper)


def get_entity_mapper(
    client_mapper: ClientMapper = Depends(get_client_mapper)
) -> EntityMapper:
    """Get entity mapper with all model mappings."""
    return EntityMapper(
        entity_mappings={
            Client: client_mapper.to_entity,
        }
    )


def get_unit_of_work(
    db: Database = Depends(get_database),
    entity_mapper: EntityMapper = Depends(get_entity_mapper)
) -> UnitOfWork:
    """Get Unit of Work instance."""
    return UnitOfWork(db, entity_mapper)


def get_client_service(
    repository: ClientRepository = Depends(get_client_repository),
    unit_of_work: UnitOfWork = Depends(get_unit_of_work)
) -> ClientService:
    """Get Client service instance."""
    return ClientService(repository, unit_of_work)
