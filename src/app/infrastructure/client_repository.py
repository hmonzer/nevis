from uuid import UUID
from typing import Optional
from sqlalchemy import select

from src.shared.database.base_repo import BaseRepository
from src.shared.database.database import Database
from src.app.domain.models import Client
from src.app.infrastructure.entities.client_entity import ClientEntity
from src.app.infrastructure.mappers.client_mapper import ClientMapper


class ClientRepository(BaseRepository[ClientEntity, Client]):
    """Repository for Client operations."""

    def __init__(self, db: Database, mapper: ClientMapper):
        super().__init__(db, mapper)

    async def get_by_id(self, client_id: UUID) -> Optional[Client]:
        """Get a client by ID."""
        return await self.find_one(
            select(ClientEntity).where(ClientEntity.id == client_id)
        )

    async def get_by_email(self, email: str) -> Optional[Client]:
        """Get a client by email."""
        return await self.find_one(
            select(ClientEntity).where(ClientEntity.email == email)
        )
