# AegisVision Backend

Backend for AegisVision - receives events from AI modules, stores them, manages alerts, and pushes real-time alerts.

## Setup

cd backend
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload

Server runs at http://127.0.0.1:8000
Interactive API docs: http://127.0.0.1:8000/docs

## Event Format (sent by AI modules via POST /events)

{
  "event_type": "intrusion",
  "timestamp": "2026-09-01T14:00:00",
  "source": "camera_05",
  "severity": "high",
  "data": {
    "track_id": 30,
    "zone": "restricted_area_4"
  }
}

- event_type: e.g. intrusion, vehicle_detection, human_detection, anpr, face_detection, loitering, night_movement
- timestamp: ISO 8601 format
- source: camera ID
- severity: low, medium, or high
- data: module-specific extra fields (flexible object)

Events with severity high automatically become alerts and are broadcast live over WebSocket.

## Camera Format (sent via POST /cameras)

{
  "camera_id": "camera_01",
  "location": "Main Gate",
  "status": "active"
}

- camera_id: unique identifier for the camera (matches source in events)
- location: human-readable location name
- status: defaults to active if not provided

## APIs

| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Server health check |
| POST | /events | Submit a new event |
| GET | /events | List all events |
| GET | /alerts | List high-severity events only |
| PATCH | /alerts/{id}/acknowledge | Mark an alert as acknowledged |
| POST | /cameras | Register a new camera |
| GET | /cameras | List all registered cameras |
| WS | /ws/alerts | Live WebSocket feed of high-severity alerts |

## Frontend Integration (Person 6)

Connect via WebSocket to receive alerts live using JavaScript's WebSocket API, pointing to ws://127.0.0.1:8000/ws/alerts. Each message received matches the event format above, with severity set to high.

To mark an alert as handled, send a PATCH request to /alerts/{id}/acknowledge. This sets acknowledged to true on that event.
