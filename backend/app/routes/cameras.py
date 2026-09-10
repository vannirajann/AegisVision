from fastapi import APIRouter, Depends
from sqlmodel import Session, select
from app.models.event import CameraIn
from app.database.db import CameraDB, get_session

router = APIRouter()

@router.post("/cameras")
def register_camera(camera: CameraIn, session: Session = Depends(get_session)):
    db_camera = CameraDB(**camera.dict())
    session.add(db_camera)
    session.commit()
    session.refresh(db_camera)
    return {"message": "Camera registered", "camera": db_camera}

@router.get("/cameras")
def get_cameras(session: Session = Depends(get_session)):
    cameras = session.exec(select(CameraDB)).all()
    return {"count": len(cameras), "cameras": cameras}