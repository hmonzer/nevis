from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status

from src.app.core.services.client_service import ClientService
from src.client.schemas import CreateClientRequest, ClientResponse
from src.app.api.dependencies import get_client_service
from src.app.api.mappers import to_client_response
from src.shared.exceptions import EntityNotFound, ConflictingEntityFound
from src.app.logging import get_logger

router = APIRouter(prefix="/clients", tags=["clients"])
logger = get_logger(__name__)


@router.post("/", response_model=ClientResponse, status_code=status.HTTP_201_CREATED)
async def create_client(
    request: CreateClientRequest,
    service: ClientService = Depends(get_client_service)
) -> ClientResponse:
    """Create a new client."""
    try:
        client = await service.create_client(request)
        return to_client_response(client)
    except ConflictingEntityFound as e:
        logger.error(f"Failed to create client: {e}")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except ValueError as e:
        logger.error(f"Failed to create client due to validation error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/{client_id}", response_model=ClientResponse)
async def get_client(
    client_id: UUID,
    service: ClientService = Depends(get_client_service)
) -> ClientResponse:
    """Get a client by ID."""
    try:
        client = await service.get_client(client_id)
        return to_client_response(client)
    except EntityNotFound as e:
        logger.error(f"Client not found: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
