from sqlmodel import SQLModel, Field, create_engine, Session, Column, JSON
from datetime import datetime
from typing import Dict, Any, Optional

from app.models.user import UserDB


class EventDB(SQLModel, table=True):
    __tablename__ = "events"

    id: Optional[int] = Field(default=None, primary_key=True)
    event_type: str
    timestamp: datetime
    source: str
    severity: str
    data: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON)
    )
    acknowledged: bool = False


class CameraDB(SQLModel, table=True):
    __tablename__ = "cameras"

    id: Optional[int] = Field(default=None, primary_key=True)
    camera_id: str
    location: str
    status: str = "active"


DATABASE_URL = "sqlite:///./aegisvision.db"

engine = create_engine(
    DATABASE_URL,
    echo=True
)


def init_db():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
