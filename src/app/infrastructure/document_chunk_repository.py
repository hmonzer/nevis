from uuid import UUID
from typing import Optional
from sqlalchemy import select

from src.app.core.domain.models import DocumentChunk
from src.shared.database.base_repo import BaseRepository
from src.shared.database.database import Database
from src.app.infrastructure.entities import DocumentChunkEntity
from src.app.infrastructure.mappers.document_chunk_mapper import DocumentChunkMapper


class DocumentChunkRepository(BaseRepository[DocumentChunkEntity, DocumentChunk]):
    """Repository for DocumentChunk operations."""

    def __init__(self, db: Database, mapper: DocumentChunkMapper):
        super().__init__(db, mapper)

    async def get_by_id(self, chunk_id: UUID) -> Optional[DocumentChunk]:
        """Get a document chunk by ID."""
        return await self.find_one(
            select(DocumentChunkEntity).where(DocumentChunkEntity.id == chunk_id)
        )

    async def get_by_document_id(self, document_id: UUID) -> list[DocumentChunk]:
        """Get all chunks for a specific document, ordered by chunk_index."""
        return await self.find_all(
            select(DocumentChunkEntity)
            .where(DocumentChunkEntity.document_id == document_id)
            .order_by(DocumentChunkEntity.chunk_index)
        )
