"""Mappers for converting between domain models and API schemas."""
from src.app.core.domain.models import Client, Document, SearchResult
from src.client.schemas import (
    ClientResponse,
    DocumentResponse,
    DocumentStatusEnum,
    SearchResultResponse,
    SearchResultTypeEnum,
)


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
        summary=document.summary,
        created_at=document.created_at,
    )


def to_search_result_response(result: SearchResult) -> SearchResultResponse:
    """
    Convert a SearchResult domain model to SearchResultResponse API schema.

    Args:
        result: Domain model containing either a Client or Document

    Returns:
        API response schema with the appropriate entity type
    """
    # Convert the entity based on type
    if result.type == "CLIENT":
        entity_response = to_client_response(result.entity)  # type: ignore
    else:
        entity_response = to_document_response(result.entity)  # type: ignore

    return SearchResultResponse(
        type=SearchResultTypeEnum(result.type),
        entity=entity_response,
        score=result.score,
    )
