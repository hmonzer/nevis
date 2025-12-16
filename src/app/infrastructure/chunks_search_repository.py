"""Repository for document chunk search operations (vector and keyword)."""
import re
from typing import Optional

from sqlalchemy import select, func

from src.app.core.domain.models import ChunkSearchResult, DocumentChunk
from src.shared.database.base_repo import BaseRepository
from src.shared.database.database import Database
from src.app.infrastructure.entities.document_entity import DocumentChunkEntity
from src.app.infrastructure.mappers.document_chunk_mapper import DocumentChunkMapper


class ChunksRepositorySearch(BaseRepository[DocumentChunkEntity, DocumentChunk]):
    """
    Repository for searching document chunks using vector similarity.

    This repository extends BaseRepository to provide vector search capabilities
    using pgvector's cosine distance operator for finding semantically similar chunks.
    """

    def __init__(self, db: Database, mapper: DocumentChunkMapper):
        super().__init__(db, mapper)

    async def search_by_vector(
        self,
        query_vector: list[float],
        limit: int = 10,
        similarity_threshold: Optional[float] = None
    ) -> list[ChunkSearchResult]:
        """
        Search for document chunks similar to the query vector.

        Uses pgvector's cosine distance (<=> operator) to find similar chunks.
        Results are ordered by similarity (highest first) and limited to top K.

        Args:
            query_vector: The embedding vector to search for (must be 384-dimensional)
            limit: Maximum number of results to return (default: 10)
            similarity_threshold: Optional minimum similarity score (-1.0 to 1.0).
                                If provided, only results with score >= threshold are returned.

        Returns:
            List of ChunkSearchResult objects containing chunks and their similarity scores,
            ordered by score descending (most similar first)

        Raises:
            ValueError: If query_vector is empty or has wrong dimensions
        """
        if not query_vector:
            raise ValueError("Query vector cannot be empty")

        if len(query_vector) != 384:
            raise ValueError(f"Query vector must be 384-dimensional, got {len(query_vector)}")

        # Compute similarity as 1 - cosine_distance
        similarity = (1 - DocumentChunkEntity.embedding.cosine_distance(query_vector)).label("similarity")

        # Build query with threshold filter in SQL for efficiency
        query = (
            select(DocumentChunkEntity, similarity)
            .where(DocumentChunkEntity.embedding.isnot(None))
        )

        if similarity_threshold is not None:
            query = query.where(
                (1 - DocumentChunkEntity.embedding.cosine_distance(query_vector)) >= similarity_threshold
            )

        query = (
            query
            .order_by(DocumentChunkEntity.embedding.cosine_distance(query_vector))
            .limit(limit)
        )

        results = await self._search_with_scores(query)
        return [ChunkSearchResult(chunk=chunk, score=score) for chunk, score in results]

    async def search_by_keyword(
        self,
        query_text: str,
        limit: int = 10
    ) -> list[ChunkSearchResult]:
        """
        Search for document chunks using full-text keyword search.

        Uses PostgreSQL's ts_vector and ts_query for efficient full-text search.
        Uses OR logic between words so documents matching ANY term are returned.
        Results are ranked by ts_rank (documents matching more terms rank higher).

        Args:
            query_text: The text query to search for
            limit: Maximum number of results to return (default: 10)

        Returns:
            List of ChunkSearchResult objects containing chunks and their relevance scores,
            ordered by score descending (most relevant first)

        Raises:
            ValueError: If query_text is empty
        """
        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty")

        cleaned_words = await self._generate_words_for_querying(query_text)

        # Build OR-based query: word1 | word2 | word3
        # This matches documents containing ANY of the search terms
        # ts_rank will give higher scores to documents matching more terms
        or_query = ' | '.join(cleaned_words)
        ts_query = func.to_tsquery('english', or_query)
        ts_vector = func.to_tsvector('english', DocumentChunkEntity.chunk_content)

        query = (
            select(
                DocumentChunkEntity,
                func.ts_rank(ts_vector, ts_query).label("rank")
            )
            .where(ts_vector.op('@@')(ts_query))
            .order_by(func.ts_rank(ts_vector, ts_query).desc())
            .limit(limit)
        )

        results = await self._search_with_scores(query)
        return [ChunkSearchResult(chunk=chunk, score=score) for chunk, score in results]

    @staticmethod
    async def _generate_words_for_querying(query_text):
        # Split query into words and clean each word
        # Remove special characters that could break ts_query syntax
        words = query_text.strip().split()
        cleaned_words = [
            cleaned for word in words
            if (cleaned := re.sub(r'[^\w]', '', word))
        ]
        if not cleaned_words:
            raise ValueError("Query text must contain at least one valid word")
        return cleaned_words
