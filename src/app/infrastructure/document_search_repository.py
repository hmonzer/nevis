"""Repository for document chunk vector search operations."""
from typing import Optional

from sqlalchemy import select, func

from src.app.core.domain.models import ChunkSearchResult, DocumentChunk
from src.shared.database.base_repo import BaseRepository
from src.shared.database.database import Database
from src.app.infrastructure.entities.document_entity import DocumentChunkEntity
from src.app.infrastructure.mappers.document_chunk_mapper import DocumentChunkMapper


class DocumentSearchRepository(BaseRepository[DocumentChunkEntity, DocumentChunk]):
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
