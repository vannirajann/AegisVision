# AegisVision - Complete Integration Map

## 🏗️ PROJECT ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────┐
│                    VIDEO INPUT (Webcam/File)               │
└────────────────────────────┬────────────────────────────────┘
                             │
        ┌────────────────────┴────────────────────┐
        │                                         │
        ▼                                         ▼
   ┌─────────────┐                         ┌─────────────┐
   │ DETECTION   │                         │ ANALYTICS   │
   │ (YOLO 11)   │                         │ (Multiple)  │
   ├─────────────┤                         ├─────────────┤
   │ - Objects   │                         │ - Face      │
   │ - Tracking  │                         │ - Movement  │
   │ - Classes   │                         │ - Night     │
   │ - Bboxes    │                         │ - Loitering │
   └──────┬──────┘                         └──────┬──────┘
          │                                       │
          └───────────────┬───────────────────────┘
                          │
        ┌─────────────────▼─────────────────┐
        │   TRACKING-INTRUSION PIPELINE     │
        ├───────────────────────────────────┤
        │ - Centroid Tracking               │
        │ - Line Crossing Detection         │
        │ - Zone Entry Detection            │
        │ - Intrusion Events Generation     │
        └──────────────┬──────────────────┘
                       │
        ┌──────────────▼──────────────┐
        │   EVENTS GENERATION         │
        ├──────────────────────────────┤
        │ - Event Objects Created      │
        │ - Severity Levels Assigned   │
        │ - Timestamps Added           │
        └──────────────┬───────────────┘
                       │
        ┌──────────────▼──────────────────────┐
        │   BACKEND (FastAPI)                 │
        ├──────────────────────────────────────┤
        │ - /events (POST/GET)                 │
        │ - /alerts (HIGH severity)            │
        │ - /ws/alerts (WebSocket Live)        │
        │ - Alert Storage & Management         │
        └──────────────┬───────────────────────┘
                       │
        ┌──────────────▼──────────────────────┐
        │   FRONTEND (Dashboard)               │
        ├──────────────────────────────────────┤
        │ - Real-time Alert Display            │
        │ - Evidence Images                    │
        │ - System Status                      │
        │ - Analytics Summary                  │
        └──────────────────────────────────────┘
```

---

## 📦 MODULES BREAKDOWN

### 1. **DETECTION MODULE** (`detection/`)
**Purpose:** YOLO-based real-time object detection and tracking

**Files:**
- `main.py` - Standalone detection with webcam/video
- `api.py` - FastAPI endpoint for detection

**Output:**
```
Detection Results:
├── Object Class (person, car, bicycle, etc.)
├── Tracking ID (persistent across frames)
├── Bounding Box (x1, y1, x2, y2)
├── Confidence Score
└── Alert Images (saved to alerts/ folder)
```

**Key Features:**
- YOLO11n model (5.6MB)
- Real-time tracking with ByteTrack
- Detects: person, bicycle, car, motorcycle, bus, truck
- Saves alert images when intrusions occur

---

### 2. **ANALYTICS MODULE** (`analytics/`)
**Purpose:** Specialized detection for faces, movement, night conditions, loitering

**Components:**
- `face/face_detector.py` - Face detection
- `movement/movement_detector.py` - Motion analysis
- `night/night_detector.py` - Low-light detection
- `loitering/loitering_detector.py` - Stationary person detection
- `events/event_generator.py` - Event creation
- `alerts/alert_manager.py` - Alert management
- `evidence/evidence_manager.py` - Evidence storage

**Output:**
```
Analytics Results:
├── Faces Detected
│   ├── Face Count
│   └── Face Bounding Boxes
├── Movement Detection
│   ├── Movement Status (detected/not detected)
│   └── Movement Region
├── Night Detection
│   ├── Brightness Level
│   ├── Low-Light Status
│   └── Light Status (day/night/twilight)
└── Events Generated
    ├── Event Type (night_movement, loitering, etc.)
    ├── Person ID
    └── Details Object
```

---

### 3. **TRACKING-INTRUSION MODULE** (`tracking-intrusion/`)
**Purpose:** Advanced tracking and intrusion detection logic

**Components:**
- `tracker.py` - Centroid-based tracking
- `fence.py` - Virtual line and zone logic
- `geometry.py` - Geometric calculations
- `events.py` - Intrusion event creation
- `pipeline.py` - Main orchestration

**Output:**
```
Intrusion Events:
├── Intrusion Type
│   ├── Line Crossing
│   └── Zone Entry
├── Object Info
│   ├── Track ID
│   ├── Object Type
│   └── Confidence
└── Event Metadata
    ├── Timestamp
    └── Event Details
```

---

### 4. **MAIN PIPELINE** (`pipeline/`)
**Purpose:** Orchestrates all modules into unified workflow

**File:** `surveillance_pipeline.py`

**Output:**
```
Pipeline Process Frame Output:
├── Faces
│   └── Face Detection Results
├── People
│   └── Tracked Person Objects
├── Movement
│   └── Movement Detection Status
├── Low Light
│   └── Low-Light Flag
├── Brightness
│   └── Brightness Value
├── Light Status
│   └── Day/Night/Twilight Status
├── Event
│   └── Generated Event Object
└── Alerts
    └── Alert Management Results
```

---

### 5. **BACKEND** (`backend/`)
**Purpose:** Central event storage and real-time alert streaming

**Endpoints:**
```
GET    /health              - Server health check
POST   /events              - Submit new event
GET    /events              - List all events
GET    /alerts              - List high-severity alerts only
WS     /ws/alerts           - WebSocket live alert feed
```

**Event Format:**
```json
{
  "event_type": "intrusion",
  "timestamp": "2026-08-31T12:00:00",
  "source": "camera_01",
  "severity": "high",
  "data": {
    "track_id": 20,
    "zone": "restricted_area_3",
    "object_type": "person",
    "confidence": 0.95
  }
}
```

---

### 6. **FRONTEND** (`frontend/`, `detection/static/`)
**Purpose:** User dashboard for visualization and monitoring

**Pages:**
- `index.html` - Main dashboard with alerts list
- `live.html` - Live video feed monitoring

**Output Display:**
```
Dashboard Components:
├── Header
│   ├── System Logo
│   └── System Status (🟢 ACTIVE)
├── Statistics Cards
│   ├── Total Intrusion Alerts
│   ├── System Status
│   └── AI Detection Status
├── Main Dashboard
│   ├── Live Video Feed
│   └── Recent Alerts List
│       ├── Alert Timestamp
│       ├── Alert Image
│       └── Alert Details
└── Footer
    └── System Information
```

---

### 7. **ANPR MODULE** (`anpr/`)
**Purpose:** License plate recognition (not yet integrated)

**Status:** Available for future integration

---

## 🔄 DATA FLOW EXAMPLE

### Scenario: Person walks into restricted zone

```
Frame 1: VIDEO CAPTURED
    ↓
Frame 2-5: DETECTION
    - YOLO detects person (class 0)
    - ByteTrack assigns ID #42
    - Bounding box: (100, 50, 200, 300)
    ↓
Frame 3: ANALYTICS
    - Face detection: 1 face found
    - Movement detected: YES
    - Night status: NO (daylight)
    ↓
Frame 4: TRACKING-INTRUSION
    - Centroid position: (150, 175)
    - Zone check: OUTSIDE → INSIDE
    - Event triggered: Zone Entry
    ↓
Frame 5: EVENT GENERATION
    - Event Type: "intrusion"
    - Severity: "high"
    - Track ID: 42
    - Timestamp: 2026-08-31T12:30:45
    ↓
Frame 6: ALERT MANAGEMENT
    - Alert created
    - Evidence image saved
    - Alert level: HIGH
    ↓
Frame 7: BACKEND
    - Event stored in database
    - Alert broadcast via WebSocket
    ↓
Frame 8: FRONTEND
    - Alert appears in dashboard
    - Evidence image displayed
    - Status updated: 🚨 INTRUSION DETECTED!
    ↓
Frame 9: RESPONSE
    - User notified
    - Admin can review evidence
    - System logs event
```

---

## 🎯 INTEGRATION POINTS

### Detection → Tracking-Intrusion
```python
detections = [
    {
        "class": "person",
        "confidence": 0.95,
        "bbox": {"x1": 100, "y1": 50, "x2": 200, "y2": 300}
    }
]
events = intrusion_pipeline.process_frame(detections)
```

### Tracking-Intrusion → Analytics
```python
tracked_objects = {
    42: {"class": "person", "confidence": 0.95, "center": (150, 175)}
}
faces = face_detector.detect_faces(frame)
movement = movement_detector.detect_movement(frame)
night = night_detector.detect(frame)
```

### Analytics → Event Generation
```python
if movement_detected and low_light:
    event = event_generator.create_event(
        event_type="night_movement",
        person_id=42,
        details={"brightness": 25, "light_status": "night"}
    )
```

### Event → Backend
```python
POST /events
{
    "event_type": "intrusion",
    "timestamp": "2026-08-31T12:30:45",
    "source": "camera_01",
    "severity": "high",
    "data": {...}
}
```

### Backend → Frontend (WebSocket)
```javascript
const ws = new WebSocket("ws://127.0.0.1:8000/ws/alerts");
ws.onmessage = (event) => {
    const alert = JSON.parse(event.data);
    // Update dashboard with new alert
};
```

---

## 🚀 HOW TO RUN EVERYTHING

### Option 1: Detection Only
```bash
cd detection
python main.py
```

### Option 2: Detection API + Frontend Dashboard
```bash
cd detection
python -m uvicorn api:app --reload
# Open http://127.0.0.1:8000
```

### Option 3: Full Pipeline
```bash
cd pipeline
python surveillance_pipeline.py
```

### Option 4: Backend + Frontend (Full System)
```bash
# Terminal 1: Start Backend
cd backend
python -m uvicorn app.main:app --reload

# Terminal 2: Start Detection/Analytics
cd pipeline
python surveillance_pipeline.py

# Browser: Open Frontend
http://127.0.0.1:8000
```

---

## 📊 OUTPUT SUMMARY

| Module | Input | Output | Format |
|--------|-------|--------|--------|
| Detection | Video Frame | Objects + Tracking IDs | Dict with class, bbox, confidence |
| Analytics | Frame + Detections | Faces, Movement, Night Status | Dict with detection results |
| Tracking-Intrusion | Detections | Intrusion Events | Intrusion event objects |
| Event Generator | Analytics Results | Structured Events | JSON event objects |
| Backend | Events (POST) | Stored Events + Alerts | Database + WebSocket |
| Frontend | Backend WebSocket | Dashboard Display | HTML/CSS/JS UI |

---

## 🔐 Security Features

✅ Intrusion Detection (zone/line crossing)
✅ Face Detection (person identification)
✅ Night-Time Monitoring (low-light detection)
✅ Movement Analysis (motion detection)
✅ Loitering Detection (stationary person alert)
✅ Real-Time Alerts (WebSocket streaming)
✅ Evidence Storage (frame capture & logging)
✅ Multi-Source Support (multiple cameras)

---

## 📁 File Structure

```
AegisVision/
├── detection/              # YOLO Detection + API
│   ├── main.py            # Standalone detection
│   ├── api.py             # FastAPI endpoint
│   ├── alerts/            # Alert images
│   └── static/            # Dashboard HTML/CSS/JS
├── analytics/             # Detection modules
│   ├── face/              # Face detection
│   ├── movement/          # Motion detection
│   ├── night/             # Low-light detection
│   ├── loitering/         # Stationary detection
│   ├── events/            # Event generation
│   ├── alerts/            # Alert management
│   └── evidence/          # Evidence storage
├── tracking-intrusion/    # Tracking + intrusion logic
│   ├── pipeline.py        # Main pipeline
│   ├── tracker.py         # Tracking logic
│   ├── fence.py           # Zone/line detection
│   └── events.py          # Event creation
├── pipeline/              # Main orchestrator
│   └── surveillance_pipeline.py
├── backend/               # FastAPI Backend
│   ├── app/
│   │   └── main.py       # API endpoints
│   └── requirements.txt
├── frontend/              # Frontend UI
├── anpr/                  # License plate recognition
└── yolo11n.pt            # Model file
```

---

Generated: 2026-08-31
Last Updated: Integration Complete
