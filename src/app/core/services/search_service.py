"""
Unified search service that searches across both clients and documents.

This service provides a single interface to search across multiple entity types,
merging and ranking results from different search services.
"""
import asyncio
import logging
from typing import cast

from src.app.core.domain.models import (
    Client,
    Document,
    ScoredResult,
    SearchRequest,
    SearchResult,
)
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
            "Unified search for query: '%s' (top_k=%d)",
            request.query[:100],
            request.top_k,
        )

        # Execute searches in parallel for better performance
        client_results_raw, document_results_raw = await asyncio.gather(
            self.client_search_service.search(request),
            self.document_search_service.search(request),
            return_exceptions=True,
        )

        # Build unified results from both sources
        unified_results: list[SearchResult] = []

        # Handle exceptions from either service
        if isinstance(client_results_raw, Exception):
            logger.error("Client search failed: %s", client_results_raw)
        else:
            client_results = cast(list[ScoredResult[Client]], client_results_raw)
            for result in client_results:
                unified_results.append(SearchResult(
                    type="CLIENT",
                    entity=result.item,
                    score=result.value,
                ))
        if isinstance(document_results_raw, Exception):
            logger.error("Document search failed: %s", document_results_raw)
        else:
            document_results = cast(list[ScoredResult[Document]], document_results_raw)
            for result in document_results:
                unified_results.append(SearchResult(
                    type="DOCUMENT",
                    entity=result.item,
                    score=result.value,
                ))

        # Sort by score descending and take top_k
        unified_results.sort(key=lambda r: r.score, reverse=True)
        unified_results = unified_results[:request.top_k]

        logger.info(
            "Returning %d unified results (top_k=%d)", len(unified_results), request.top_k
        )

        return unified_results
