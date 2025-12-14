from uuid import UUID
from typing import Optional
from sqlalchemy import select

from src.app.core.domain.models import Document
from src.shared.database.base_repo import BaseRepository
from src.shared.database.database import Database
from src.app.infrastructure.entities.document_entity import DocumentEntity
from src.app.infrastructure.mappers.document_mapper import DocumentMapper


class DocumentRepository(BaseRepository[DocumentEntity, Document]):
    """Repository for Document operations."""

    def __init__(self, db: Database, mapper: DocumentMapper):
        super().__init__(db, mapper)

    async def get_by_id(self, document_id: UUID) -> Optional[Document]:
        """Get a document by ID."""
        return await self.find_one(
            select(DocumentEntity).where(DocumentEntity.id == document_id)
        )

    async def get_client_document_by_id(
        self, document_id: UUID, client_id: UUID
    ) -> Optional[Document]:
        """Get a document by its ID and client ID."""
        return await self.find_one(
            select(DocumentEntity).where(
                DocumentEntity.id == document_id,
                DocumentEntity.client_id == client_id
            )
        )

    async def get_by_client_id(self, client_id: UUID) -> list[Document]:
        """Get all documents for a specific client."""
        return await self.find_all(
            select(DocumentEntity).where(DocumentEntity.client_id == client_id)
        )
