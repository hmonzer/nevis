"""Mappers for converting between domain models and API schemas."""
from src.app.core.domain.models import Client
from src.client.schemas import ClientResponse


def to_client_response(client: Client) -> ClientResponse:
    """
    Convert a Client domain model to ClientResponse API schema.

    Args:
        client: Domain model

    Returns:
        API response schema
    """
    return ClientResponse(
        id=client.id,
        first_name=client.first_name,
        last_name=client.last_name,
        email=client.email,
        description=client.description,
        created_at=client.created_at,
    )
