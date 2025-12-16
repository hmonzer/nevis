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

    def __init__(
        self,
        search_repository: ClientSearchRepository,
        pg_trgm_threshold: float,
    ):
        """
        Initialize the client search service.

        Args:
            search_repository: Repository for performing client searches
            pg_trgm_threshold: Minimum trigram similarity score (0.0 to 1.0) for results.
                      Configured via client_search_trgm_threshold in settings.
        """
        self.search_repository = search_repository
        self.pg_trgm_threshold = pg_trgm_threshold

    async def search(
        self,
        request: SearchRequest
    ) -> list[ClientSearchResult]:
        """
        Search for clients matching the given query.

        Performs fuzzy text matching across client fields (email, first name,
        last name, description) and returns results ranked by relevance.

        Args:
            request: SearchRequest containing query and top_k parameters.

        Returns:
            List of ClientSearchResult objects ordered by relevance (highest similarity first)
        """
        # Perform search using repository with configured threshold
        results = await self.search_repository.search(
            query=request.query,
            threshold=self.pg_trgm_threshold,
            limit=request.top_k
        )

        return results
