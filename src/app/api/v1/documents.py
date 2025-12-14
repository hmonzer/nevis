from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status

from src.app.core.services.document_service import DocumentService
from src.client.schemas import CreateDocumentRequest, DocumentResponse
from src.app.api.dependencies import get_document_service
from src.app.api.mappers import to_document_response

router = APIRouter(prefix="/clients/{client_id}/documents", tags=["documents"])


@router.post("/", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def create_document(
    client_id: UUID,
    request: CreateDocumentRequest,
    service: DocumentService = Depends(get_document_service)
) -> DocumentResponse:
    """
    Upload and process a document for a client.

    This endpoint:
    1. Verifies the client exists
    2. Uploads content to S3
    3. Creates a document record with PENDING status
    4. Chunks the content using the configured chunking strategy
    5. Persists document chunks to the database
    6. Updates document status to PROCESSED

    Args:
        client_id: UUID of the client who owns this document
        request: Document creation request with title and content
        service: Document service (injected)

    Returns:
        DocumentResponse with created document details

    Raises:
        HTTPException 404: If client not found
        HTTPException 400: If document processing fails
    """
    try:
        document = await service.upload_document(
            client_id=client_id,
            title=request.title,
            content=request.content
        )
        return to_document_response(document)
    except ValueError as e:
        error_message = str(e)
        # Distinguish between not found and other errors
        if "not found" in error_message.lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_message
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_message
        )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    client_id: UUID,
    document_id: UUID,
    service: DocumentService = Depends(get_document_service)
) -> DocumentResponse:
    """
    Get a document by ID.

    Args:
        client_id: UUID of the client (for route consistency)
        document_id: UUID of the document to retrieve
        service: Document service (injected)

    Returns:
        DocumentResponse with document details

    Raises:
        HTTPException 404: If document not found
    """
    try:
        # TODO: Modify the query to fetch document by id and client_id (since it's already in the API request).
        document = await service.get_document(document_id)

        # Verify document belongs to the specified client
        if document.client_id != client_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {document_id} not found for client {client_id}"
            )

        return to_document_response(document)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@router.get("/", response_model=list[DocumentResponse])
async def list_client_documents(
    client_id: UUID,
    service: DocumentService = Depends(get_document_service)
) -> list[DocumentResponse]:
    """
    Get all documents for a specific client.

    Args:
        client_id: UUID of the client
        service: Document service (injected)

    Returns:
        List of DocumentResponse objects

    Raises:
        HTTPException 404: If client not found
    """
    try:
        documents = await service.get_client_documents(client_id)
        return [to_document_response(doc) for doc in documents]
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
