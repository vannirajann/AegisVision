from sqlmodel import SQLModel, Field, create_engine, Session, Column, JSON
from datetime import datetime
from typing import Dict, Any, Optional

class EventDB(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    event_type: str
    timestamp: datetime
    source: str
    severity: str
    data: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    acknowledged: bool = False

class CameraDB(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    camera_id: str
    location: str
    status: str = "active"

class UserDB(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str
    hashed_password: str
    full_name: str
    role: str = "operator"

DATABASE_URL = "sqlite:///./aegisvision.db"
engine = create_engine(DATABASE_URL, echo=True)

def init_db():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session