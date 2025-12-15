"""Service for searching client records using fuzzy text matching."""

from src.app.core.domain.models import ClientSearchResult, SearchRequest
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
        request: SearchRequest
    ) -> list[ClientSearchResult]:
        """
        Search for clients matching the given query.

        Performs fuzzy text matching across client fields (email, first name,
        last name, description) and returns results ranked by relevance.

        Args:
            request: SearchRequest containing query, top_k, and threshold parameters.
                    Validation is handled by the SearchRequest model.

        Returns:
            List of ClientSearchResult objects ordered by relevance (highest similarity first)
        """
        # Perform search using repository
        results = await self.search_repository.search(
            query=request.query,
            threshold=request.threshold,
            limit=request.top_k
        )

        return results
