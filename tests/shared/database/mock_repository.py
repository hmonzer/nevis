from uuid import UUID
from typing import Optional
from sqlalchemy import select

from src.shared.database.base_repo import BaseRepository
from tests.shared.database.mock_entities import UserEntity


class UserRepository(BaseRepository):
    async def get_by_id(self, user_id: UUID) -> Optional[UserEntity]:
        async with self.db.session_maker() as session:
            result = await session.execute(
                select(UserEntity).where(UserEntity.id == user_id)
            )
            return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> Optional[UserEntity]:
        async with self.db.session_maker() as session:
            result = await session.execute(
                select(UserEntity).where(UserEntity.email == email)
            )
            return result.scalar_one_or_none()

    async def get_all(self) -> list[UserEntity]:
        async with self.db.session_maker() as session:
            result = await session.execute(select(UserEntity))
            return list(result.scalars().all())
