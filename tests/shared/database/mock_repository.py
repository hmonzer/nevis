from uuid import UUID
from typing import Optional
from sqlalchemy import select

from src.shared.database.base_repo import BaseRepository
from src.shared.database.database import Database
from tests.shared.database.mock_entities import UserEntity, UserModel, UserMapper


class UserRepository(BaseRepository[UserEntity, UserModel]):
    def __init__(self, db: Database, mapper: UserMapper):
        super().__init__(db, mapper)

    async def get_by_id(self, user_id: UUID) -> Optional[UserModel]:
        return await self.find_one(
            select(UserEntity).where(UserEntity.id == user_id)
        )

    async def get_by_email(self, email: str) -> Optional[UserModel]:
        return await self.find_one(
            select(UserEntity).where(UserEntity.email == email)
        )

    async def get_all(self) -> list[UserModel]:
        return await self.find_all(select(UserEntity))