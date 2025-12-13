from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from src.shared.database.database import Database
from src.shared.database.entity_mapper import EntityMapper


class UnitOfWork:
    def __init__(
        self,
        db: Database,
        entity_mapper: EntityMapper,
    ) -> None:
        self.db = db
        self.session: AsyncSession
        self.entity_mapper = entity_mapper

    async def __aenter__(self):
        self.session = self.db.session_maker()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            await self.commit()
        else:
            await self.rollback()
        if self.session:
            await self.session.close()

    def _map_to_entity(self, model_instance: Any):
        return self.entity_mapper.map_to_entity(model_instance)

    def add(self, model_instance: Any):
        entity = self._map_to_entity(model_instance)
        self.session.add(entity)

    async def update(self, model_instance: Any):
        entity = self._map_to_entity(model_instance)
        await self.session.merge(entity)

    def delete(self, model_instance: Any):
        entity = self._map_to_entity(model_instance)
        self.session.delete(entity)

    async def commit(self):
        try:
            await self.session.commit()
        except Exception as e:
            await self.rollback()
            raise e

    async def rollback(self):
        await self.session.rollback()
