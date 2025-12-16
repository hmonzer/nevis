from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from dependency_injector.wiring import Provide, inject

from src.app.containers import Container
from src.app.core.services.document_service import DocumentService
from src.client.schemas import CreateDocumentRequest, DocumentResponse
from src.app.api.mappers import to_document_response
from src.shared.exceptions import EntityNotFound
from src.app.logging import get_logger

router = APIRouter(prefix="/clients/{client_id}/documents", tags=["documents"])
logger = get_logger(__name__)


@router.post("/", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
@inject
async def create_document(
    client_id: UUID,
    request: CreateDocumentRequest,
    background_tasks: BackgroundTasks,
    service: DocumentService = Depends(Provide[Container.document_service]),
) -> DocumentResponse:
    """
    Upload and process a document for a client.

    This endpoint:
    1. Verifies the client exists
    2. Uploads content to S3
    3. Creates a document record with PENDING status
    4. Schedules a background task for chunking and processing

    Args:
        client_id: UUID of the client who owns this document
        request: Document creation request with title and content
        background_tasks: FastAPI background task runner
        service: Document service (injected)

    Returns:
        DocumentResponse with created document details

    Raises:
        HTTPException 404: If client not found
        HTTPException 400: If document creation fails
    """
    try:
        document = await service.create_document(
            client_id=client_id, title=request.title, content=request.content
        )
    except EntityNotFound as e:
        logger.error(f"Failed to create document, client not found: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        logger.error(f"Failed to create document: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        )

    # Schedule the document processing to run in the background
    background_tasks.add_task(service.process_document, document.id, request.content)

    return to_document_response(document)


@router.get("/{document_id}", response_model=DocumentResponse)
@inject
async def get_document(
    client_id: UUID,
    document_id: UUID,
    service: DocumentService = Depends(Provide[Container.document_service]),
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
        document = await service.get_document_by_id_and_client_id(
            document_id, client_id
        )
        return to_document_response(document)
    except EntityNotFound as e:
        logger.error(f"Document not found for client {client_id}: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get("/", response_model=list[DocumentResponse])
@inject
async def list_client_documents(
    client_id: UUID,
    service: DocumentService = Depends(Provide[Container.document_service]),
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
    except EntityNotFound as e:
        logger.error(f"Client not found when listing documents: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
