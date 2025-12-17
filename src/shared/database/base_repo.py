import abc
from typing import Generic, TypeVar, Optional

from sqlalchemy import Executable

from src.shared.database.base_mapper import BaseEntityMapper
from src.shared.database.database import Database


TEntity = TypeVar("TEntity")
TModel = TypeVar("TModel")


class BaseRepository(abc.ABC, Generic[TEntity, TModel]):
    def __init__(self, db: Database, mapper: BaseEntityMapper[TModel, TEntity]):
        self.db = db
        self.mapper = mapper

    async def find_one(self, statement: Executable) -> Optional[TModel]:
        async with self.db.session_maker() as session:
            result = await session.execute(statement)
            entity = result.scalar_one_or_none()
            if entity is None:
                return None
            return self.mapper.to_model(entity)

    async def find_all(self, statement: Executable) -> list[TModel]:
        async with self.db.session_maker() as session:
            result = await session.execute(statement)
            entities = list(result.scalars().all())
            return [self.mapper.to_model(entity) for entity in entities]

    async def _search_with_scores(
        self, statement: Executable
    ) -> list[tuple[TModel, float]]:
        """
        Execute a query returning (entity, score) tuples and map entities to models.

        Use this for search queries that return relevance/similarity scores alongside entities.

        Args:
            statement: SQLAlchemy select statement returning (Entity, score) tuples

        Returns:
            List of (model, score) tuples
        """
        async with self.db.session_maker() as session:
            result = await session.execute(statement)
            rows = result.all()

        return [(self.mapper.to_model(entity), float(score)) for entity, score in rows]
