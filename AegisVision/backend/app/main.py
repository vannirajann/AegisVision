from fastapi import FastAPI
from app.routes import health, events, alerts
from app.database.db import init_db

app = FastAPI(title="AegisVision Backend")

@app.on_event("startup")
def on_startup():
    init_db()

app.include_router(health.router)
app.include_router(events.router)
app.include_router(alerts.router)