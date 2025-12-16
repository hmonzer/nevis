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

        # Build the query using cosine distance
        # Note: cosine distance = 1 - cosine similarity
        # So we convert: similarity = 1 - distance
        query = (
            select(
                DocumentChunkEntity,
                (1 - DocumentChunkEntity.embedding.cosine_distance(query_vector)).label("similarity")
            )
            .where(DocumentChunkEntity.embedding.isnot(None))  # Only search chunks with embeddings
            .order_by(DocumentChunkEntity.embedding.cosine_distance(query_vector))  # Closest first
            .limit(limit)
        )

        # Execute query
        async with self.db.session_maker() as session:
            result = await session.execute(query)
            rows = result.all()

        # Convert to ChunkSearchResult objects
        search_results = []
        for entity, similarity_score in rows:
            # Apply threshold filter if specified
            if similarity_threshold is not None and similarity_score < similarity_threshold:
                continue

            chunk = self.mapper.to_model(entity)
            search_results.append(
                ChunkSearchResult(
                    chunk=chunk,
                    score=float(similarity_score)
                )
            )

        return search_results

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

        # Split query into words and clean each word
        # Remove special characters that could break ts_query syntax
        words = query_text.strip().split()
        cleaned_words = []
        for word in words:
            # Keep only alphanumeric characters
            cleaned = re.sub(r'[^\w]', '', word)
            if cleaned:
                cleaned_words.append(cleaned)

        if not cleaned_words:
            raise ValueError("Query text must contain at least one valid word")

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

        # Execute query
        async with self.db.session_maker() as session:
            result = await session.execute(query)
            rows = result.all()

        # Convert to ChunkSearchResult objects
        search_results = []
        for entity, rank_score in rows:
            chunk = self.mapper.to_model(entity)
            search_results.append(
                ChunkSearchResult(
                    chunk=chunk,
                    score=float(rank_score)
                )
            )

        return search_results
