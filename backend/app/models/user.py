from typing import Optional
from sqlmodel import SQLModel, Field


class UserDB(SQLModel, table=True):
    __tablename__ = "userdb"

    id: Optional[int] = Field(default=None, primary_key=True)
    username: str
    hashed_password: str
    full_name: str
    role: str = "operator"
