from datetime import datetime
from enum import Enum as PyEnum
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import String, DateTime, func, Enum, ForeignKey, Integer, Text, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.shared.database.database import Base


class DocumentStatus(str, PyEnum):
    """Document processing status enum."""
    PENDING = "PENDING"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"


class DocumentEntity(Base):
    """SQLAlchemy model for Document table."""
    __tablename__ = "documents"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    client_id: Mapped[UUID] = mapped_column(ForeignKey("clients.id"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255))
    s3_key: Mapped[str] = mapped_column(String(512), nullable=False)
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus, values_callable=lambda x: [e.value for e in x]),
        default=DocumentStatus.PENDING,
        nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    # Relationship to chunks
    chunks: Mapped[list["DocumentChunkEntity"]] = relationship(
        "DocumentChunkEntity",
        back_populates="document",
        cascade="all, delete-orphan"
    )


class DocumentChunkEntity(Base):
    """SQLAlchemy model for DocumentChunk table."""
    __tablename__ = "document_chunks"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_content: Mapped[str] = mapped_column(Text, nullable=False)

    # Vector embedding field - nullable for now
    # Using all-MiniLM-L6-v2 which produces 384-dimensional embeddings
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(384),  # Dimension for all-MiniLM-L6-v2 embeddings
        nullable=True
    )

    # Relationship to document
    document: Mapped[DocumentEntity] = relationship(
        "DocumentEntity",
        back_populates="chunks"
    )


# HNSW index for fast approximate nearest neighbor search on embeddings
# This dramatically improves vector similarity search performance (O(log n) vs O(n))
# Using cosine distance operator (<=>) which matches DocumentSearchRepository queries
Index(
    'ix_document_chunks_embedding_hnsw',
    DocumentChunkEntity.embedding,
    postgresql_using='hnsw',
    postgresql_with={'m': 16, 'ef_construction': 64},
    postgresql_ops={'embedding': 'vector_cosine_ops'}
)
