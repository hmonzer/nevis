"""Service for searching client records using fuzzy text matching."""
import logging
from typing import Optional

from src.app.core.domain.models import ClientSearchResult, SearchRequest
from src.app.core.services.reranker import RerankerService
from src.app.infrastructure.client_search_repository import ClientSearchRepository

logger = logging.getLogger(__name__)


def _extract_client_content(result: ClientSearchResult) -> str:
    """
    Extract searchable text content from a client for reranking.

    Combines client name and description into a single text representation
    that the CrossEncoder can score against the query.

    Args:
        result: ClientSearchResult containing the client to extract content from

    Returns:
        Text content representing the client
    """
    client = result.client
    parts = [f"Client Name: {client.first_name} {client.last_name}","Email Address: {client.email}"]
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
    """

    def __init__(
        self,
        search_repository: ClientSearchRepository,
        pg_trgm_threshold: float,
        reranker_service: Optional[RerankerService] = None,
        reranker_score_threshold: float = 2.0,
        retrieval_multiplier: int = 3,
    ):
        """
        Initialize the client search service.

        Args:
            search_repository: Repository for performing client searches
            pg_trgm_threshold: Minimum trigram similarity score (0.0 to 1.0) for results.
                      Configured via client_search.trgm_threshold in settings.
            reranker_service: Optional service for reranking results using CrossEncoder.
                      When provided, results are reranked for better relevance scoring.
            reranker_score_threshold: Minimum CrossEncoder score for results.
                      Results below this threshold are filtered out.
                      Only applied when reranker_service is provided.
            retrieval_multiplier: Multiplier for top_k when reranking is enabled.
                      Fetches more candidates to ensure good results after reranking.
        """
        self.search_repository = search_repository
        self.pg_trgm_threshold = pg_trgm_threshold
        self.reranker_service = reranker_service
        self.reranker_score_threshold = reranker_score_threshold
        self.retrieval_multiplier = retrieval_multiplier

    async def search(
        self,
        request: SearchRequest
    ) -> list[ClientSearchResult]:
        """
        Search for clients matching the given query.

        Performs fuzzy text matching across client fields (email, first name,
        last name, description). When a reranker is configured, results are
        reranked using CrossEncoder for improved relevance scoring.

        Args:
            request: SearchRequest containing query and top_k parameters.

        Returns:
            List of ClientSearchResult objects ordered by relevance.
            Scores are trigram similarity (0-1) without reranker,
            or CrossEncoder logits (~-12 to +12) with reranker.
        """
        # Fetch more candidates if reranking is enabled
        retrieval_limit = request.top_k
        if self.reranker_service:
            retrieval_limit = request.top_k * self.retrieval_multiplier

        logger.info(
            "Client search for query: '%s' (top_k=%d, retrieval_limit=%d)",
            request.query,
            request.top_k,
            retrieval_limit
        )

        # Perform search using repository with configured threshold
        candidates = await self.search_repository.search(
            query=request.query,
            threshold=self.pg_trgm_threshold,
            limit=retrieval_limit
        )

        logger.info("pg_trgm search returned %d candidates", len(candidates))
        for c in candidates:
            logger.debug("  Candidate: %s %s (score=%.4f)", c.client.first_name, c.client.last_name, c.score)

        if not candidates:
            return []

        # Apply reranking if reranker is available
        if self.reranker_service:
            logger.info("Applying reranking to %d client candidates", len(candidates))
            ranked = await self.reranker_service.rerank(
                query=request.query,
                items=candidates,
                content_extractor=_extract_client_content,
                top_k=request.top_k
            )
            # Convert RankedResult back to ClientSearchResult
            results = [
                ClientSearchResult(client=r.item.client, score=r.score)
                for r in ranked
            ]
            logger.info("Reranking complete. Returning %d results", len(results))
            for r in results:
                logger.info("  Reranked: %s %s (score=%.4f)", r.client.first_name, r.client.last_name, r.score)

            # Filter by score threshold
            results = self._filter_by_threshold(results)
        else:
            # No reranker - just return top_k from candidates
            results = candidates[:request.top_k]

        return results

    def _filter_by_threshold(
        self,
        results: list[ClientSearchResult]
    ) -> list[ClientSearchResult]:
        """
        Filter results below the reranker score threshold.

        Args:
            results: List of ClientSearchResult with CrossEncoder scores

        Returns:
            Filtered list with only results above threshold
        """
        pre_filter_count = len(results)
        filtered = [r for r in results if r.score >= self.reranker_score_threshold]

        if pre_filter_count != len(filtered):
            logger.info(
                "Filtered %d client results below threshold %.2f (kept %d)",
                pre_filter_count - len(filtered),
                self.reranker_score_threshold,
                len(filtered)
            )

        return filtered
