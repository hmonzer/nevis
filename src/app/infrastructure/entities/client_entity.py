from datetime import datetime
from uuid import UUID, uuid4
from sqlalchemy import String, Text, DateTime, func, Index
from sqlalchemy.orm import Mapped, mapped_column

from src.shared.database.database import Base


class ClientEntity(Base):
    """SQLAlchemy model for Client table."""
    __tablename__ = "clients"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    first_name: Mapped[str] = mapped_column(String(100), index=True)
    last_name: Mapped[str] = mapped_column(String(100), index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )


# GIN indexes for fuzzy text search using pg_trgm extension
# These significantly speed up similarity searches in ClientSearchRepository
Index(
    'ix_clients_first_name_gin',
    ClientEntity.first_name,
    postgresql_using='gin',
    postgresql_ops={'first_name': 'gin_trgm_ops'}
)

Index(
    'ix_clients_last_name_gin',
    ClientEntity.last_name,
    postgresql_using='gin',
    postgresql_ops={'last_name': 'gin_trgm_ops'}
)

Index(
    'ix_clients_description_gin',
    ClientEntity.description,
    postgresql_using='gin',
    postgresql_ops={'description': 'gin_trgm_ops'}
)
