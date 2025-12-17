"""Service for searching client records using fuzzy text matching."""
import logging
from typing import Optional

from src.app.config import ClientSearchSettings
from src.app.core.domain.models import Client, ScoredResult, SearchRequest
from src.app.core.services.reranker import RerankerService
from src.app.infrastructure.client_search_repository import ClientSearchRepository

logger = logging.getLogger(__name__)


def _extract_client_content(client: Client) -> str:
    """
    Extract searchable text content from a client for reranking.

    Combines client name, email, and description into a single text representation
    that the CrossEncoder can score against the query.

    Args:
        client: Client to extract content from

    Returns:
        Text content representing the client
    """
    parts = [f"Client Name: {client.first_name} {client.last_name}", f"Email Address: {client.email}"]
    if client.description:
        parts.append(f"Client Description: {client.description}")
    return ". ".join(parts)


class ClientSearchService:
    """
    Service for searching clients using fuzzy text matching and optional reranking.

    This service provides a high-level interface for searching clients
    across multiple fields (email, name, description) using PostgreSQL's
    pg_trgm fuzzy matching extension, with optional CrossEncoder reranking
    for improved relevance scoring.

    Returns ScoredResult[Client] with full score history tracking.
    """

    def __init__(
        self,
        search_repository: ClientSearchRepository,
        settings: ClientSearchSettings,
        reranker_service: Optional[RerankerService] = None,
    ):
        """
        Initialize the client search service.

        Args:
            search_repository: Repository for performing client searches
            settings: Client search settings including thresholds and multipliers
            reranker_service: Optional service for reranking results using CrossEncoder.
                      When provided, results are reranked for better relevance scoring.
        """
        self.search_repository = search_repository
        self.settings = settings
        self.reranker_service = reranker_service

    async def search(
        self,
        request: SearchRequest
    ) -> list[ScoredResult[Client]]:
        """
        Search for clients matching the given query.

        Performs fuzzy text matching across client fields (email, first name,
        last name, description). When a reranker is configured, results are
        reranked using CrossEncoder for improved relevance scoring.

        Args:
            request: SearchRequest containing query and top_k parameters.

        Returns:
            List of ScoredResult[Client] with full score history.
            Score history tracks: trigram similarity → reranking (if enabled).
        """
        # Fetch more candidates if reranking is enabled
        retrieval_limit = request.top_k
        if self.reranker_service:
            retrieval_limit = request.top_k * self.settings.retrieval_multiplier

        logger.info(
            "Client search for query: '%s' (top_k=%d, retrieval_limit=%d)",
            request.query,
            request.top_k,
            retrieval_limit
        )

        # Perform search using repository with configured threshold
        candidates = await self.search_repository.search(
            query=request.query,
            threshold=self.settings.trgm_threshold,
            limit=retrieval_limit
        )

        logger.info("pg_trgm search returned %d candidates", len(candidates))
        if not candidates:
            return []

        # Apply reranking if reranker is available
        if self.reranker_service:
            logger.info("Applying reranking to %d client candidates", len(candidates))
            # Reranker uses assign_score() to preserve history - no unwrapping needed!
            results = await self.reranker_service.rerank(
                query=request.query,
                results=candidates,
                content_extractor=_extract_client_content,  # Extract from Client directly
                top_k=request.top_k
            )
            logger.info("Reranking complete. Returning %d results", len(results))
            # Filter by score threshold
            results = ScoredResult.filter_by_threshold(results, self.settings.reranker_score_threshold)
        else:
            # No reranker - just return top_k from candidates
            results = candidates[:request.top_k]

        return results
