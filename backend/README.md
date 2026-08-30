\# AegisVision Backend



Backend for AegisVision — receives events from AI modules, stores them, and pushes real-time alerts.



\## Setup



cd backend

venv\\Scripts\\activate

pip install -r requirements.txt

uvicorn app.main:app --reload



Server runs at http://127.0.0.1:8000

Interactive API docs: http://127.0.0.1:8000/docs



\## Event Format (sent by AI modules via POST /events)



{

&#x20; "event\_type": "intrusion",

&#x20; "timestamp": "2026-08-30T17:35:00",

&#x20; "source": "camera\_04",

&#x20; "severity": "high",

&#x20; "data": {

&#x20;   "track\_id": 20,

&#x20;   "zone": "restricted\_area\_3"

&#x20; }

}



\- event\_type: e.g. "intrusion", "vehicle\_detection", "human\_detection", "anpr", "face\_detection", "loitering", "night\_movement"

\- timestamp: ISO 8601 format

\- source: camera ID

\- severity: "low" | "medium" | "high"

\- data: module-specific extra fields (flexible object)



\## APIs



| Method | Path | Description |

|--------|------|-------------|

| GET | /health | Server health check |

| POST | /events | Submit a new event |

| GET | /events | List all events |

| GET | /alerts | List high-severity events only |

| WS | /ws/alerts | Live WebSocket feed of high-severity alerts |



\## Frontend Integration (Person 6)



Connect via WebSocket to receive alerts live:



const ws = new WebSocket("ws://127.0.0.1:8000/ws/alerts");

ws.onmessage = (event) => {

&#x20; const alert = JSON.parse(event.data);

&#x20; // handle alert in UI

};



Each message received matches the event format above, with severity: "high".

