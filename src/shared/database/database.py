import logging

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base

logger = logging.getLogger(__name__)

Base = declarative_base()

class DatabaseSettings(BaseModel):
    db_url: str


class Database:
    def __init__(self, db_settings: DatabaseSettings) -> None:
        self._engine = create_async_engine(db_settings.db_url)
        self.session_maker: async_sessionmaker[AsyncSession] = async_sessionmaker(self._engine, expire_on_commit=False)