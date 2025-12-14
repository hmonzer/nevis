from src.shared.database.base_mapper import BaseEntityMapper
from src.app.core.domain.models import DocumentChunk
from src.app.infrastructure.entities import DocumentChunkEntity


class DocumentChunkMapper(BaseEntityMapper[DocumentChunk, DocumentChunkEntity]):
    """Mapper for converting between DocumentChunk domain model and DocumentChunkEntity."""

    @staticmethod
    def to_entity(model_instance: DocumentChunk) -> DocumentChunkEntity:
        """Convert DocumentChunk (domain model) to DocumentChunkEntity (database entity)."""
        return DocumentChunkEntity(
            id=model_instance.id,
            document_id=model_instance.document_id,
            chunk_index=model_instance.chunk_index,
            chunk_content=model_instance.chunk_content,
            embedding=model_instance.embedding,
        )

    @staticmethod
    def to_model(entity: DocumentChunkEntity) -> DocumentChunk:
        """Convert DocumentChunkEntity (database entity) to DocumentChunk (domain model)."""
        return DocumentChunk(
            id=entity.id,
            document_id=entity.document_id,
            chunk_index=entity.chunk_index,
            chunk_content=entity.chunk_content,
            embedding=entity.embedding,
        )
