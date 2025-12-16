from src.shared.database.base_mapper import BaseEntityMapper
from src.app.core.domain.models import Document, DocumentStatus
from src.app.infrastructure.entities.document_entity import DocumentEntity, DocumentStatus as EntityDocumentStatus


class DocumentMapper(BaseEntityMapper[Document, DocumentEntity]):
    """Mapper for converting between Document domain model and DocumentEntity."""

    @staticmethod
    def to_entity(model_instance: Document) -> DocumentEntity:
        """Convert Document (domain model) to DocumentEntity (database entity)."""
        return DocumentEntity(
            id=model_instance.id,
            client_id=model_instance.client_id,
            title=model_instance.title,
            s3_key=model_instance.s3_key,
            status=EntityDocumentStatus(model_instance.status.value),
            summary=model_instance.summary,
            created_at=model_instance.created_at,
        )

    @staticmethod
    def to_model(entity: DocumentEntity) -> Document:
        """Convert DocumentEntity (database entity) to Document (domain model)."""
        return Document(
            id=entity.id,
            client_id=entity.client_id,
            title=entity.title,
            s3_key=entity.s3_key,
            status=DocumentStatus(entity.status.value),
            summary=entity.summary,
            created_at=entity.created_at,
        )
