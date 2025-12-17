"""Service for searching documents using semantic vector search aggregated from chunks."""
import logging
from uuid import UUID

from src.app.core.domain.models import Document, ScoredResult, Score, SearchRequest, DocumentChunk
from src.app.core.services.chunks_search_service import DocumentChunkSearchService
from src.app.infrastructure.document_repository import DocumentRepository

logger = logging.getLogger(__name__)


class DocumentSearchService:
    """
    Service for semantic search at document level.

    This service aggregates chunk-level search results to return document-level results.
    It uses the chunk search service to find relevant chunks, then groups them by document
    and assigns each document the highest relevance score from its matching chunks.

    Returns ScoredResult[Document] preserving the score from the best-matching chunk.
    """

    def __init__(
        self,
        chunk_search_service: DocumentChunkSearchService,
        document_repository: DocumentRepository,
        chunk_retrieval_multiplier: int = 5,
    ):
        """
        Initialize the document search service.

        Args:
            chunk_search_service: Service for searching document chunks
            document_repository: Repository for retrieving full document records
            chunk_retrieval_multiplier: Multiplier for top_k to determine how many
                chunks to fetch per requested document.
        """
        self.chunk_search_service = chunk_search_service
        self.document_repository = document_repository
        self.chunk_retrieval_multiplier = chunk_retrieval_multiplier

    async def search(
        self,
        request: SearchRequest
    ) -> list[ScoredResult[Document]]:
        """
        Search for documents semantically similar to the query.

        This method:
        1. Searches for relevant chunks using the chunk search service
        2. Groups chunks by their parent document
        3. Assigns each document the highest score from its matching chunks
        4. Returns top K documents ranked by relevance score (descending)

        Args:
            request: SearchRequest containing query, top_k, and threshold parameters.
                    Validation is handled by the SearchRequest model.

        Returns:
            List of ScoredResult[Document] preserving the score and source
            from the best-matching chunk for each document.
        """
        logger.info("Searching for documents matching query: '%s' (top_k=%d)", request.query[:100], request.top_k)

        # Search for relevant chunks
        # Fetch more chunks to ensure we get enough documents
        chunk_limit = request.top_k * self.chunk_retrieval_multiplier
        chunk_search_request = SearchRequest(
            query=request.query,
            top_k=chunk_limit,
        )
        chunk_results = await self.chunk_search_service.search(chunk_search_request)

        logger.info("Retrieved %d chunks from chunk search", len(chunk_results))

        if not chunk_results:
            logger.info("No chunks found matching query")
            return []

        # Group chunks by document ID and track best score (with source)
        best_chunk_scores: dict[UUID, ScoredResult[DocumentChunk]] = {}
        for chunk_result in chunk_results:
            doc_id = chunk_result.item.document_id
            current_best = best_chunk_scores.get(doc_id)
            # Keep the chunk result with the highest score for this document
            if current_best is None or chunk_result.value > current_best.value:
                best_chunk_scores[doc_id] = chunk_result

        logger.info("Found %d unique documents from chunks", len(best_chunk_scores))

        # Sort documents by score (descending) and take top_k
        sorted_doc_ids = sorted(
            best_chunk_scores.keys(),
            key=lambda doc_id: best_chunk_scores[doc_id].value,
            reverse=True
        )[:request.top_k]

        # Retrieve full document records in batch
        documents = await self.document_repository.get_by_ids(sorted_doc_ids)

        # Create a mapping of document ID to document for quick lookup
        doc_map = {doc.id: doc for doc in documents}

        # Build results maintaining the sort order
        # Use the score and source from the best-matching chunk
        results: list[ScoredResult[Document]] = []
        for doc_id in sorted_doc_ids:
            document = doc_map.get(doc_id)
            if document is not None:
                best_chunk = best_chunk_scores[doc_id]
                results.append(ScoredResult(
                    item=document,
                    score=Score(value=best_chunk.value, source=best_chunk.source),
                    score_history=best_chunk.score_history  # Preserve history from best chunk
                ))
            else:
                logger.warning("Document %s not found in repository", doc_id)

        logger.info("Returning %d document results", len(results))

        return results
