from uuid import UUID
from typing import Optional
from sqlalchemy import select, func, or_, case

from src.app.core.domain.models import Client
from src.shared.database.base_repo import BaseRepository
from src.shared.database.database import Database

from src.app.infrastructure.entities.client_entity import ClientEntity
from src.app.infrastructure.mappers.client_mapper import ClientMapper


class ClientRepository(BaseRepository[ClientEntity, Client]):
    """Repository for Client operations."""

    def __init__(self, db: Database, mapper: ClientMapper):
        super().__init__(db, mapper)

    async def get_by_id(self, client_id: UUID) -> Optional[Client]:
        """Get a client by ID."""
        return await self.find_one(
            select(ClientEntity).where(ClientEntity.id == client_id)
        )

    async def get_by_email(self, email: str) -> Optional[Client]:
        """Get a client by email."""
        return await self.find_one(
            select(ClientEntity).where(ClientEntity.email == email)
        )

    async def search(self, query: str, threshold: float = 0.1) -> list[Client]:
        """
        Search for clients using fuzzy matching across email, first name, last name, and description.

        Uses PostgreSQL's pg_trgm extension for fuzzy text matching. Searches across:
        - email
        - first_name
        - last_name
        - description (if not null)

        Args:
            query: Search term to match against client fields
            threshold: Minimum similarity score (0.0 to 1.0). Default is 0.1.
                      Lower values return more results with less strict matching.

        Returns:
            List of clients ordered by relevance (highest similarity first)
        """
        # Calculate similarity scores for each field
        email_similarity = func.similarity(ClientEntity.email, query)
        first_name_similarity = func.similarity(ClientEntity.first_name, query)
        last_name_similarity = func.similarity(ClientEntity.last_name, query)
        # Handle null descriptions by using 0.0 similarity for null values
        description_similarity = case(
            (ClientEntity.description.is_(None), 0.0),
            else_=func.similarity(ClientEntity.description, query)
        )

        # Maximum similarity across all fields
        max_similarity = func.greatest(
            email_similarity,
            first_name_similarity,
            last_name_similarity,
            description_similarity
        ).label("max_similarity")

        # Build query with similarity filtering and ordering
        stmt = (
            select(ClientEntity)
            .where(
                or_(
                    email_similarity > threshold,
                    first_name_similarity > threshold,
                    last_name_similarity > threshold,
                    description_similarity > threshold
                )
            )
            .order_by(max_similarity.desc())
        )

        return await self.find_all(stmt)
