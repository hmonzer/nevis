"""Mappers for converting between domain models and API schemas."""
from src.app.core.domain.models import Client, Document
from src.client.schemas import ClientResponse, DocumentResponse, DocumentStatusEnum


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


def to_document_response(document: Document) -> DocumentResponse:
    """
    Convert a Document domain model to DocumentResponse API schema.

    Args:
        document: Domain model

    Returns:
        API response schema
    """
    return DocumentResponse(
        id=document.id,
        client_id=document.client_id,
        title=document.title,
        s3_key=document.s3_key,
        status=DocumentStatusEnum(document.status.value),
        created_at=document.created_at,
    )
