import abc

from src.shared.database.database import Database


class BaseRepository(abc.ABC):
    def __init__(self, db: Database):
        self.db = db