from pydantic import BaseModel
from datetime import datetime
from typing import Dict, Any

class EventIn(BaseModel):
    event_type: str
    timestamp: datetime
    source: str
    severity: str
    data: Dict[str, Any] = {}