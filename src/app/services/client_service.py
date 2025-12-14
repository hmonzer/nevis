from uuid import UUID, uuid4
from datetime import datetime, UTC

from sqlalchemy.exc import IntegrityError

from src.shared.database.unit_of_work import UnitOfWork
from src.client.schemas import CreateClientRequest
from src.app.domain.models import Client
from src.app.infrastructure.client_repository import ClientRepository


class ClientService:
    """Service for handling Client business logic."""

    def __init__(self, repository: ClientRepository, unit_of_work: UnitOfWork):
        self.repository = repository
        self.unit_of_work = unit_of_work

    async def create_client(self, request: CreateClientRequest) -> Client:
        """Create a new client."""
        # Create domain model with generated ID and timestamp
        client = Client(
            id=uuid4(),
            first_name=request.first_name,
            last_name=request.last_name,
            email=request.email,
            description=request.description,
            created_at=datetime.now(UTC),
        )

        # Persist using unit of work - database will enforce email uniqueness
        try:
            async with self.unit_of_work:
                self.unit_of_work.add(client)
        except IntegrityError as e:
            # Database constraint violation (e.g., duplicate email)
            raise ValueError(f"Client with email {request.email} already exists") from e

        return client

    async def get_client(self, client_id: UUID) -> Client:
        """Get a client by ID."""
        client = await self.repository.get_by_id(client_id)
        if not client:
            raise ValueError(f"Client with ID {client_id} not found")
        return client

    async def get_client_by_email(self, email: str) -> Client:
        """Get a client by email."""
        client = await self.repository.get_by_email(email)
        if not client:
            raise ValueError(f"Client with email {email} not found")
        return client
