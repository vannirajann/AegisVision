from fastapi import APIRouter, Depends
from sqlmodel import Session, select
from app.models.event import EventIn
from app.database.db import EventDB, get_session
from app.websocket.manager import manager

router = APIRouter()

@router.post("/events")
async def receive_event(event: EventIn, session: Session = Depends(get_session)):
    db_event = EventDB(**event.dict())
    session.add(db_event)
    session.commit()
    session.refresh(db_event)

    print(f"Saved event to database: {db_event}")

    if db_event.severity == "high":
        print(f"ALERT triggered: {db_event}")
        await manager.broadcast({
            "event_type": db_event.event_type,
            "timestamp": db_event.timestamp.isoformat(),
            "source": db_event.source,
            "severity": db_event.severity,
            "data": db_event.data,
        })

    return {"message": "Event received", "event": db_event}

@router.get("/events")
def get_events(session: Session = Depends(get_session)):
    events = session.exec(select(EventDB)).all()
    return {"count": len(events), "events": events}