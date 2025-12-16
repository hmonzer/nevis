"""
Unified search service that searches across both clients and documents.

This service provides a single interface to search across multiple entity types,
merging and ranking results from different search services.
"""
import asyncio
import logging
from typing import Any, cast

from src.app.core.domain.models import SearchRequest, SearchResult, ClientSearchResult, DocumentSearchResult
from src.app.core.services.client_search_service import ClientSearchService
from src.app.core.services.document_search_service import DocumentSearchService

logger = logging.getLogger(__name__)


class SearchService:
    """
    Unified search service that queries both clients and documents.

    Combines results from ClientSearchService and DocumentSearchService,
    ranks them by relevance score, and returns a unified result set.
    """

    def __init__(
        self,
        client_search_service: ClientSearchService,
        document_search_service: DocumentSearchService,
    ):
        """
        Initialize the unified search service.

        Args:
            client_search_service: Service for searching clients
            document_search_service: Service for searching documents
        """
        self.client_search_service = client_search_service
        self.document_search_service = document_search_service

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        """
        Search across both clients and documents.

        Executes searches in parallel, combines results, sorts by score descending,
        assigns ranks, and returns the top_k results.

        Args:
            request: Search request with query, top_k, and threshold

        Returns:
            List of SearchResult objects sorted by score descending, limited to top_k
        """
        logger.info(
            "Unified search for query: '%s' (top_k=%d, threshold=%.2f)",
            request.query[:100],
            request.top_k,
            request.threshold,
        )

        # Execute searches in parallel for better performance
        client_results_raw, document_results_raw = await asyncio.gather(
            self.client_search_service.search(request),
            self.document_search_service.search(request),
            return_exceptions=True,
        )

        # Handle exceptions from either service and ensure we have lists
        client_results: list[ClientSearchResult] = []
        if isinstance(client_results_raw, Exception):
            logger.error("Client search failed: %s", client_results_raw)
        else:
            client_results = cast(list[ClientSearchResult], client_results_raw)

        document_results: list[DocumentSearchResult] = []
        if isinstance(document_results_raw, Exception):
            logger.error("Document search failed: %s", document_results_raw)
        else:
            document_results = cast(list[DocumentSearchResult], document_results_raw)

        logger.info(
            "Retrieved %d clients and %d documents",
            len(client_results),
            len(document_results),
        )

        # Collect all results with their metadata
        all_results: list[tuple[str, Any, float]] = []

        for client_result in client_results:
            all_results.append(("CLIENT", client_result.client, client_result.score))

        for doc_result in document_results:
            all_results.append(("DOCUMENT", doc_result.document, doc_result.score))

        # Sort by score descending
        all_results.sort(key=lambda x: x[2], reverse=True)

        # Limit to top_k results
        all_results = all_results[: request.top_k]

        # Create SearchResult objects with proper ranks (1-based)
        unified_results: list[SearchResult] = []
        for idx, (entity_type, entity, score) in enumerate(all_results, start=1):
            unified_results.append(
                SearchResult(
                    type=entity_type,  # type: ignore
                    entity=entity,
                    score=score,
                    rank=idx,
                )
            )

        logger.info(
            "Returning %d unified results (top_k=%d)", len(unified_results), request.top_k
        )

        return unified_results
