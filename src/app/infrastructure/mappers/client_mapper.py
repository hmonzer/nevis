from pydantic.v1 import EmailStr

from src.shared.database.base_mapper import BaseEntityMapper
from src.app.core.domain.models import Client
from src.app.infrastructure.entities.client_entity import ClientEntity


class ClientMapper(BaseEntityMapper[Client, ClientEntity]):
    """Mapper for converting between Client domain model and ClientEntity."""

    @staticmethod
    def to_entity(model_instance: Client) -> ClientEntity:
        """Convert a Client (domain model) to ClientEntity (database entity)."""
        return ClientEntity(
            id=model_instance.id,
            first_name=model_instance.first_name,
            last_name=model_instance.last_name,
            email=model_instance.email.__str__(),
            description=model_instance.description,
            created_at=model_instance.created_at,
        )

    @staticmethod
    def to_model(entity: ClientEntity) -> Client:
        """Convert a ClientEntity (database entity) to Client (domain model)."""
        return Client(
            id=entity.id,
            first_name=entity.first_name,
            last_name=entity.last_name,
            email=EmailStr(entity.email),
            description=entity.description,
            created_at=entity.created_at,
        )
