from pydantic import BaseModel
from datetime import datetime
from typing import Dict, Any, Optional

class EventIn(BaseModel):
    event_type: str
    timestamp: datetime
    source: str
    severity: str
    data: Dict[str, Any] = {}

class CameraIn(BaseModel):
    camera_id: str
    location: str
    status: Optional[str] = "active"