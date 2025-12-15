"""Service for searching documents using semantic vector search aggregated from chunks."""
import logging
from collections import defaultdict
from uuid import UUID

from src.app.core.domain.models import DocumentSearchResult
from src.app.core.services.chunks_search_service import DocumentChunkSearchService
from src.app.infrastructure.document_repository import DocumentRepository

logger = logging.getLogger(__name__)


class DocumentSearchService:
    """
    Service for semantic search at document level.

    This service aggregates chunk-level search results to return document-level results.
    It uses the chunk search service to find relevant chunks, then groups them by document
    and assigns each document the highest relevance score from its matching chunks.
    """

    def __init__(
        self,
        chunk_search_service: DocumentChunkSearchService,
        document_repository: DocumentRepository
    ):
        """
        Initialize the document search service.

        Args:
            chunk_search_service: Service for searching document chunks
            document_repository: Repository for retrieving full document records
        """
        self.chunk_search_service = chunk_search_service
        self.document_repository = document_repository

    async def search(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.5
    ) -> list[DocumentSearchResult]:
        """
        Search for documents semantically similar to the query.

        This method:
        1. Searches for relevant chunks using the chunk search service
        2. Groups chunks by their parent document
        3. Assigns each document the highest score from its matching chunks
        4. Returns top K documents ranked by relevance score (descending)

        Args:
            query: The search query string
            top_k: Maximum number of documents to return (default: 10)
            threshold: Minimum similarity score for chunks (default: 0.5)

        Returns:
            List of DocumentSearchResult objects containing documents and their relevance scores,
            ordered by score descending (most relevant first)

        Raises:
            ValueError: If query is empty, threshold is invalid, or top_k is invalid
        """
        # Validate inputs
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if threshold < -1.0 or threshold > 1.0:
            raise ValueError("threshold must be between -1.0 and 1.0")

        logger.info("Searching for documents matching query: '%s' (top_k=%d)", query[:100], top_k)

        # Search for relevant chunks
        # Fetch more chunks to ensure we get enough documents
        chunk_limit = top_k * 5
        chunk_results = await self.chunk_search_service.search(
            query=query,
            top_k=chunk_limit,
            similarity_threshold=threshold
        )

        logger.info("Retrieved %d chunks from chunk search", len(chunk_results))

        if not chunk_results:
            logger.info("No chunks found matching query")
            return []

        # Group chunks by document ID and track highest score
        document_scores: dict[UUID, float] = {}
        for chunk_result in chunk_results:
            doc_id = chunk_result.chunk.document_id
            current_score = document_scores.get(doc_id, float('-inf'))
            # Keep the highest score for this document
            document_scores[doc_id] = max(current_score, chunk_result.score)

        logger.info("Found %d unique documents from chunks", len(document_scores))

        # Sort documents by score (descending) and take top_k
        sorted_doc_ids = sorted(
            document_scores.keys(),
            key=lambda doc_id: document_scores[doc_id],
            reverse=True
        )[:top_k]

        # Retrieve full document records in batch
        documents = await self.document_repository.get_by_ids(sorted_doc_ids)

        # Create a mapping of document ID to document for quick lookup
        doc_map = {doc.id: doc for doc in documents}

        # Build results maintaining the sort order
        results = []
        for doc_id in sorted_doc_ids:
            document = doc_map.get(doc_id)
            if document:
                results.append(DocumentSearchResult(
                    document=document,
                    score=document_scores[doc_id]
                ))
            else:
                logger.warning("Document %s not found in repository", doc_id)

        logger.info("Returning %d document results", len(results))

        return results
