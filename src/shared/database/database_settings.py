from pydantic import BaseModel


class DatabaseSettings(BaseModel):
    db_url: str
