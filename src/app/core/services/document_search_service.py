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

    async def search(self, request: SearchRequest) -> list[ScoredResult[Document]]:
        """
        Search for documents semantically similar to the query.

        Args:
            request: SearchRequest containing query and top_k parameters.

        Returns:
            List of ScoredResult[Document] ranked by relevance score descending.
        """
        logger.info("Searching for documents: '%s' (top_k=%d)", request.query[:100], request.top_k)

        chunk_results = await self._search_document_chunks(request)
        if not chunk_results:
            return []

        best_chunks_by_doc = self._group_chunks_by_document(chunk_results)
        top_ranking_doc_ids = sorted(
            best_chunks_by_doc.keys(),
            key=lambda doc_id: best_chunks_by_doc[doc_id].value,
            reverse=True
        )[:request.top_k]
        documents = await self.document_repository.get_by_ids(top_ranking_doc_ids)
        return self._build_results(top_ranking_doc_ids, documents, best_chunks_by_doc)

    async def _search_document_chunks(self, request: SearchRequest) -> list[ScoredResult[DocumentChunk]]:
        """Fetch relevant chunks from chunk search service."""
        chunk_limit = request.top_k * self.chunk_retrieval_multiplier
        chunk_request = SearchRequest(query=request.query, top_k=chunk_limit)
        results = await self.chunk_search_service.search(chunk_request)
        logger.info("Retrieved %d chunks", len(results))
        return results

    @staticmethod
    def _group_chunks_by_document(chunk_results: list[ScoredResult[DocumentChunk]]) -> dict[
        UUID, ScoredResult[DocumentChunk]]:
        """Group chunks by document ID, keeping only the best-scoring chunk per document."""
        best_by_doc: dict[UUID, ScoredResult[DocumentChunk]] = {}
        for chunk_result in chunk_results:
            doc_id = chunk_result.item.document_id
            current_best = best_by_doc.get(doc_id)
            if current_best is None or chunk_result.value > current_best.value:
                best_by_doc[doc_id] = chunk_result
        logger.info("Found %d unique documents", len(best_by_doc))
        return best_by_doc

    @staticmethod
    def _build_results(doc_ids: list[UUID], documents: list[Document],
                       best_chunks_by_doc: dict[UUID, ScoredResult[DocumentChunk]] ) -> list[ScoredResult[Document]]:
        """Build final results preserving sort order and chunk scores."""
        doc_map = {doc.id: doc for doc in documents}
        results: list[ScoredResult[Document]] = []

        for doc_id in doc_ids:
            document = doc_map.get(doc_id)
            if document is None:
                logger.warning("Document %s not found", doc_id)
                continue

            best_chunk = best_chunks_by_doc[doc_id]
            results.append(ScoredResult(
                item=document,
                score=Score(value=best_chunk.value, source=best_chunk.source),
                score_history=best_chunk.score_history,
            ))

        logger.info("Returning %d document results", len(results))
        return results
