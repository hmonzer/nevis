"""Service for searching document chunks using semantic vector search."""
import logging
from typing import Optional

from src.app.core.domain.models import ChunkSearchResult, SearchRequest
from src.app.core.services.embedding import EmbeddingService
from src.app.core.services.reranker import RerankerService
from src.app.infrastructure.document_search_repository import DocumentSearchRepository

logger = logging.getLogger(__name__)


class DocumentChunkSearchService:
    """
    Service for semantic search across document chunks.

    This service combines:
    1. Embedding generation to convert queries to vectors
    2. Vector similarity search to find candidate chunks
    3. Optional reranking to refine results using cross-encoder models
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        search_repository: DocumentSearchRepository,
        reranker_service: Optional[RerankerService] = None
    ):
        """
        Initialize the document search service.

        Args:
            embedding_service: Service for generating text embeddings
            search_repository: Repository for vector similarity search
            reranker_service: Optional service for reranking results
        """
        self.embedding_service = embedding_service
        self.search_repository = search_repository
        self.reranker_service = reranker_service

    async def search(
        self,
        request: SearchRequest
    ) -> list[ChunkSearchResult]:
        """
        Search for document chunks semantically similar to the query.

        This method:
        1. Converts the query string to an embedding vector
        2. Searches for chunks with similar embeddings (retrieval phase)
        3. Optionally reranks results using cross-encoder for better accuracy
        4. Returns top K results ranked by relevance score (descending)

        Args:
            request: SearchRequest containing query, top_k, and threshold parameters.
                    Validation is handled by the SearchRequest model.

        Returns:
            List of ChunkSearchResult objects containing chunks and their relevance scores,
            ordered by score descending (most relevant first).
            Scores are cosine similarity if no reranker, or cross-encoder logits if reranked.
        """
        logger.info("Searching for chunks similar to query: '%s' (top_k=%d)", request.query[:100], request.top_k)

        # Convert query to embedding vector
        embedding_result = await self.embedding_service.embed_query(request.query)
        logger.debug("Generated query embedding with %d dimensions", len(embedding_result.embedding))

        # Search for similar chunks (retrieval phase)
        # Fetch more candidates if we're going to rerank
        retrieval_limit = request.top_k * 3 if self.reranker_service else request.top_k

        results = await self.search_repository.search_by_vector(
            query_vector=embedding_result.embedding,
            limit=retrieval_limit,
            similarity_threshold=request.threshold
        )

        logger.info("Retrieved %d candidate chunks from vector search", len(results))

        # Apply reranking if reranker is available
        if self.reranker_service and results:
            logger.info("Applying reranking to %d candidates", len(results))
            results = await self.reranker_service.rerank(request.query, results, top_k=request.top_k)
            logger.info("Reranking complete. Returning top %d results", len(results))
        else:
            # No reranker - just return top_k from vector search
            results = results[:request.top_k]

        logger.info("Search complete. Returning %d results", len(results))

        return results
