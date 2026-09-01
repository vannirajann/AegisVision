from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import health, events, alerts, cameras
from app.database.db import init_db

app = FastAPI(title="AegisVision Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    init_db()

app.include_router(health.router)
app.include_router(events.router)
app.include_router(alerts.router)
app.include_router(cameras.router)