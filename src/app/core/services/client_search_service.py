"""Service for searching client records using fuzzy text matching."""

from src.app.core.domain.models import ClientSearchResult
from src.app.infrastructure.client_search_repository import ClientSearchRepository


class ClientSearchService:
    """
    Service for searching clients using fuzzy text matching.

    This service provides a high-level interface for searching clients
    across multiple fields (email, name, description) using PostgreSQL's
    pg_trgm fuzzy matching extension.
    """

    def __init__(self, search_repository: ClientSearchRepository):
        """
        Initialize the client search service.

        Args:
            search_repository: Repository for performing client searches
        """
        self.search_repository = search_repository

    async def search(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.1
    ) -> list[ClientSearchResult]:
        """
        Search for clients matching the given query.

        Performs fuzzy text matching across client fields (email, first name,
        last name, description) and returns results ranked by relevance.

        Args:
            query: Search term to match against client fields
            top_k: Maximum number of results to return (default: 10)
            threshold: Minimum similarity score (0.0 to 1.0, default: 0.1).
                      Lower values return more results with less strict matching.

        Returns:
            List of ClientSearchResult objects ordered by relevance (highest similarity first)

        Raises:
            ValueError: If query is empty, threshold is out of range, or top_k is invalid
        """
        # Validate query
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        # Validate threshold
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")

        # Validate top_k
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        # Perform search using repository
        results = await self.search_repository.search(
            query=query,
            threshold=threshold,
            limit=top_k
        )

        return results
