from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from sqlmodel import Session, select
from app.database.db import EventDB, get_session
from app.websocket.manager import manager

router = APIRouter()

@router.get("/alerts")
def get_alerts(session: Session = Depends(get_session)):
    alerts = session.exec(select(EventDB).where(EventDB.severity == "high")).all()
    return {"count": len(alerts), "alerts": alerts}

@router.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # keeps the connection alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)