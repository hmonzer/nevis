"""Repository for client fuzzy search operations."""
from sqlalchemy import select, func, or_, case

from src.app.core.domain.models import Client, ScoredResult, Score, ScoreSource
from src.shared.database.base_repo import BaseRepository
from src.shared.database.database import Database
from src.app.infrastructure.entities.client_entity import ClientEntity
from src.app.infrastructure.mappers.client_mapper import ClientMapper


class ClientSearchRepository(BaseRepository[ClientEntity, Client]):
    """
    Repository for fuzzy searching clients using PostgreSQL's pg_trgm extension.

    This repository provides fuzzy text matching across client fields (email, name, description)
    and returns results ranked by relevance.
    """

    def __init__(self, db: Database, mapper: ClientMapper):
        super().__init__(db, mapper)

    async def search(self, query: str, threshold: float = 0.1, limit: int | None = None) -> list[ScoredResult[Client]]:
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
            limit: Maximum number of results to return. If None, returns all matching results.

        Returns:
            List of ScoredResult[Client] with TRIGRAM_SIMILARITY source,
            ordered by relevance (highest similarity first)

        Raises:
            ValueError: If query is empty or threshold is out of range
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")

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

        # Build query with similarity filtering (threshold applied in SQL) and ordering
        stmt = (
            select(ClientEntity, max_similarity)
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

        if limit is not None:
            stmt = stmt.limit(limit)

        results = await self._search_with_scores(stmt)
        return [
            ScoredResult(
                item=client,
                score=Score(value=score, source=ScoreSource.TRIGRAM_SIMILARITY)
            )
            for client, score in results
        ]
