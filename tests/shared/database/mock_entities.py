from uuid import UUID
from pydantic import BaseModel
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from src.shared.database.base_mapper import BaseEntityMapper
from src.shared.database.database import Base
from src.shared.database.entity_mapper import EntityMapper


class UserModel(BaseModel):
    id: UUID
    name: str
    email: str


class UserEntity(Base):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(100))


class UserMapper(BaseEntityMapper[UserModel, UserEntity]):
    """Example mapper that converts between UserModel and UserEntity."""

    @staticmethod
    def to_entity(model_instance: UserModel) -> UserEntity:
        """Convert a UserModel (domain model) to UserEntity (database entity)."""
        return UserEntity(
            id=model_instance.id,
            name=model_instance.name,
            email=model_instance.email,
        )

    @staticmethod
    def to_model(entity: UserEntity) -> UserModel:
        """Convert a UserEntity (database entity) to UserModel (domain model)."""
        return UserModel(
            id=entity.id,
            name=entity.name,
            email=entity.email,
        )


def get_test_entity_mapper() -> EntityMapper:
    return EntityMapper(
        entity_mappings={
            UserModel: UserMapper.to_entity,
        }
    )
