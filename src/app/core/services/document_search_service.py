"""Service for searching document chunks using semantic vector search."""
import logging
from typing import Optional

from src.app.core.domain.models import ChunkSearchResult
from src.app.core.services.embedding import EmbeddingService
from src.app.infrastructure.document_search_repository import DocumentSearchRepository

logger = logging.getLogger(__name__)


class DocumentChunkSearchService:
    """
    Service for semantic search across document chunks.

    This service combines embedding generation with vector similarity search
    to find document chunks that are semantically similar to a query string.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        search_repository: DocumentSearchRepository
    ):
        """
        Initialize the document search service.

        Args:
            embedding_service: Service for generating text embeddings
            search_repository: Repository for vector similarity search
        """
        self.embedding_service = embedding_service
        self.search_repository = search_repository

    async def search(
        self,
        query: str,
        top_k: int = 10,
        similarity_threshold: float = 0.5
    ) -> list[ChunkSearchResult]:
        """
        Search for document chunks semantically similar to the query.

        This method:
        1. Converts the query string to an embedding vector
        2. Searches for chunks with similar embeddings
        3. Returns top K results ranked by similarity score (descending)

        Args:
            query: The search query string
            top_k: Maximum number of results to return (default: 10)
            similarity_threshold: Optional minimum similarity score (-1.0 to 1.0).
                                If provided, only results with score >= threshold are returned.

        Returns:
            List of ChunkSearchResult objects containing chunks and their similarity scores,
            ordered by score descending (most relevant first)

        Raises:
            ValueError: If query is empty or invalid, or if top_k < 1
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        if similarity_threshold is not None and (similarity_threshold < -1.0 or similarity_threshold > 1.0):
            raise ValueError("similarity_threshold must be between -1.0 and 1.0")

        logger.info("Searching for chunks similar to query: '%s' (top_k=%d)", query[:100], top_k)

        # Convert query to embedding vector
        embedding_result = await self.embedding_service.embed_query(query)
        logger.debug("Generated query embedding with %d dimensions", len(embedding_result.embedding))

        # Search for similar chunks
        results = await self.search_repository.search_by_vector(
            query_vector=embedding_result.embedding,
            limit=top_k,
            similarity_threshold=similarity_threshold
        )

        logger.info("Found %d matching chunks", len(results))

        return results
