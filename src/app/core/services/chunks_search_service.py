"""Service for searching document chunks using hybrid search (vector + keyword)."""
import asyncio
import logging
from typing import Optional

from src.app.core.domain.models import ChunkSearchResult, SearchRequest
from src.app.core.services.embedding import EmbeddingService
from src.app.core.services.reranker import RerankerService
from src.app.core.services.rrf import ReciprocalRankFusion
from src.app.infrastructure.chunks_search_repository import ChunksRepositorySearch

logger = logging.getLogger(__name__)


class DocumentChunkSearchService:
    """
    Service for hybrid search across document chunks.

    This service combines:
    1. Vector similarity search (semantic understanding)
    2. Keyword search using PostgreSQL full-text search (lexical matching)
    3. Reciprocal Rank Fusion (RRF) to combine results from both search methods
    4. Optional reranking to refine results using cross-encoder models
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        search_repository: ChunksRepositorySearch,
        rrf: ReciprocalRankFusion,
        reranker_service: Optional[RerankerService] = None
    ):
        """
        Initialize the document search service.

        Args:
            embedding_service: Service for generating text embeddings
            search_repository: Repository for vector and keyword search
            rrf: Reciprocal Rank Fusion instance for combining search results
            reranker_service: Optional service for reranking results
        """
        self.embedding_service = embedding_service
        self.search_repository = search_repository
        self.rrf = rrf
        self.reranker_service = reranker_service

    async def search(
        self,
        request: SearchRequest
    ) -> list[ChunkSearchResult]:
        """
        Search for document chunks using hybrid search (vector + keyword).

        This method:
        1. Runs vector search and keyword search in parallel
        2. Combines results using Reciprocal Rank Fusion (RRF)
        3. Optionally reranks fused results using cross-encoder for better accuracy
        4. Returns top K results ranked by relevance score (descending)

        Args:
            request: SearchRequest containing query, top_k, and threshold parameters.
                    Validation is handled by the SearchRequest model.

        Returns:
            List of ChunkSearchResult objects containing chunks and their relevance scores,
            ordered by score descending (most relevant first).
            Scores are RRF fusion scores if no reranker, or cross-encoder logits if reranked.
        """
        logger.info("Hybrid search for query: '%s' (top_k=%d)", request.query[:100], request.top_k)

        # Fetch more candidates for fusion and potential reranking
        retrieval_limit = request.top_k * 3 if self.reranker_service else request.top_k * 2

        # Run vector search and keyword search in parallel
        embedding_task = self.embedding_service.embed_query(request.query)
        keyword_task = self.search_repository.search_by_keyword(
            query_text=request.query,
            limit=retrieval_limit
        )

        # Wait for embedding first, then run vector search
        embedding_result, keyword_results = await asyncio.gather(
            embedding_task,
            keyword_task
        )

        logger.debug("Generated query embedding with %d dimensions", len(embedding_result.embedding))
        logger.info("Keyword search returned %d results", len(keyword_results))

        # Run vector search with the embedding
        vector_results = await self.search_repository.search_by_vector(
            query_vector=embedding_result.embedding,
            limit=retrieval_limit,
            similarity_threshold=request.threshold
        )

        logger.info("Vector search returned %d results", len(vector_results))

        # Fuse results using RRF
        fused_results = self.rrf.fuse(vector_results, keyword_results)
        logger.info("RRF fusion produced %d unique results", len(fused_results))

        # Apply reranking if reranker is available
        if self.reranker_service and fused_results:
            logger.info("Applying reranking to %d candidates", len(fused_results))
            results = await self.reranker_service.rerank(
                request.query,
                fused_results[:retrieval_limit],  # Limit candidates for reranking
                top_k=request.top_k
            )
            logger.info("Reranking complete. Returning top %d results", len(results))
        else:
            # No reranker - just return top_k from fused results
            results = fused_results[:request.top_k]

        logger.info("Hybrid search complete. Returning %d results", len(results))

        return results
