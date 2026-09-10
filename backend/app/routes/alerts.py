from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, HTTPException
from sqlmodel import Session, select
from app.database.db import EventDB, get_session
from app.websocket.manager import manager
from app.auth import get_current_user

router = APIRouter()

@router.get("/alerts")
def get_alerts(session: Session = Depends(get_session)):
    alerts = session.exec(select(EventDB).where(EventDB.severity == "high")).all()
    return {"count": len(alerts), "alerts": alerts}

@router.patch("/alerts/{alert_id}/acknowledge")
def acknowledge_alert(
    alert_id: int,
    session: Session = Depends(get_session),
    current_user = Depends(get_current_user)
):
    alert = session.get(EventDB, alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    alert.acknowledged = True
    session.add(alert)
    session.commit()
    session.refresh(alert)

    return {
        "message": "Alert acknowledged",
        "acknowledged_by": current_user.full_name,
        "alert": alert
    }

@router.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)