from uuid import UUID
from pydantic import BaseModel
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

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


def map_user_model_to_entity(user_model: UserModel) -> UserEntity:
    return UserEntity(
        id=user_model.id,
        name=user_model.name,
        email=user_model.email,
    )


def get_test_entity_mapper() -> EntityMapper:
    return EntityMapper(
        entity_mappings={
            UserModel: map_user_model_to_entity,
        }
    )
