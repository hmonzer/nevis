from typing import Any

from sqlalchemy.orm import Session

from src.shared.database.database import Database
from src.shared.database.entity_mapper import EntityMapper


class UnitOfWork:
    def __init__(
        self,
        db: Database,
        entity_mapper: EntityMapper,
    ) -> None:
        self.db = db
        self.session: Session
        self.entity_mapper = entity_mapper

    def __enter__(self):
        with self.db.session_maker() as self.session:
            # self.session.expire_on_commit = False
            return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
        if self.session:
            self.session.close()

    def _map_to_entity(self, model_instance: Any):
        return self.entity_mapper.map_to_entity(model_instance)

    def add(self, model_instance: Any):
        entity = self._map_to_entity(model_instance)
        self.session.add(entity)

    def update(self, model_instance: Any):
        entity = self._map_to_entity(model_instance)
        self.session.merge(entity)

    def delete(self, model_instance: Any):
        entity = self._map_to_entity(model_instance)
        self.session.delete(entity)

    def commit(self):
        try:
            self.session.commit()
        except Exception as e:
            self.rollback()
            raise e

    def rollback(self):
        self.session.rollback()
