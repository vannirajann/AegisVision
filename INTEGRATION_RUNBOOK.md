# AegisVision - Complete Integration Guide

## 🚀 RUNNING THE COMPLETE SYSTEM

### Prerequisites
```bash
# Install all dependencies
pip install opencv-python ultralytics fastapi uvicorn
pip install -r backend/requirements.txt
```

---

## 📋 OPTION 1: Standalone Detection Only

**What it does:**
- Runs YOLO detection on webcam/video
- Detects intrusions
- Saves alert images locally

**How to run:**
```bash
cd detection
python main.py
```

**Output:**
```
🚨 INTRUSION DETECTED!
Object: PERSON
ID: 42
Evidence: alerts/intruder_person_ID_42_20260831_123045_123456.jpg
```

**Files created:**
- `detection/alerts/intruder_*.jpg` - Evidence images
- Console logs of detections

---

## 📋 OPTION 2: Detection API + Web Dashboard

**What it does:**
- Runs YOLO detection on video file
- Serves FastAPI backend with endpoints
- Displays live dashboard in browser

**How to run:**
```bash
cd detection
python -m uvicorn api:app --reload --port 8000
```

**Output:**
- Terminal: `Uvicorn running on http://127.0.0.1:8000`
- Browser: Open `http://127.0.0.1:8000`

**Available Endpoints:**
- `GET http://127.0.0.1:8000/` - Main dashboard
- `GET http://127.0.0.1:8000/alerts` - JSON alerts list
- `GET http://127.0.0.1:8000/alert-images/{filename}` - View alert image
- `GET http://127.0.0.1:8000/live` - Live monitoring page

**Dashboard Features:**
- Real-time stats (alert count, system status)
- Live video feed
- Alert list with images
- Responsive design

---

## 📋 OPTION 3: Full Pipeline (Analytics + Tracking + Events)

**What it does:**
- Combines all analytics modules (face, movement, night detection)
- Runs tracking-intrusion pipeline
- Generates structured events
- Displays console output

**How to run:**
```bash
cd pipeline
python surveillance_pipeline.py
```

**Output - Console:**
```
Initializing AegisVision pipeline...
All AegisVision modules loaded successfully
Video source initialized successfully
Surveillance pipeline ready

EVENT GENERATED: {
    "event_id": "EVT_20260831_123045_001",
    "event_type": "night_movement",
    "timestamp": "2026-08-31T12:30:45",
    "severity": "HIGH",
    "person_id": 42,
    "details": {
        "message": "Night-time movement detected",
        "people_count": 1,
        "brightness": 32,
        "light_status": "night"
    }
}

ALERT GENERATED: {
    "alert_id": "ALT_20260831_123045_001",
    "event_type": "night_movement",
    "severity": "HIGH",
    "level": 8
}

EVIDENCE SAVED: evidence_data/evidence_1234.json
```

**Output - Files Created:**
- `evidence_data/evidence_*.json` - Evidence records

---

## 📋 OPTION 4: Backend API Server Only

**What it does:**
- Runs FastAPI backend
- Provides REST API for event management
- Streams alerts via WebSocket
- Stores events in memory

**How to run:**
```bash
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

**Output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     API docs at http://127.0.0.1:8000/docs
```

**Available Endpoints:**
- `GET /health` - Health check
- `POST /events` - Submit event
- `GET /events` - List all events
- `GET /alerts` - List high-severity alerts only
- `WS /ws/alerts` - WebSocket alert feed

**Example API Usage:**
```bash
# Health check
curl http://127.0.0.1:8000/health

# Get all events
curl http://127.0.0.1:8000/events

# Get high-severity alerts
curl http://127.0.0.1:8000/alerts

# Submit new event
curl -X POST http://127.0.0.1:8000/events \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "intrusion",
    "timestamp": "2026-08-31T12:30:45",
    "source": "camera_01",
    "severity": "high",
    "data": {
      "track_id": 42,
      "zone": "restricted_area_3"
    }
  }'
```

---

## 📋 OPTION 5: Complete Integrated System (RECOMMENDED)

**What it does:**
- Runs detection in background
- Runs analytics + tracking pipeline
- Sends events to backend API
- Serves frontend dashboard
- Real-time WebSocket alerts

**How to run:**

### Terminal 1 - Backend API
```bash
cd backend
python -m uvicorn app.main:app --reload --port 8000
```
**Output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Terminal 2 - Main Pipeline
```bash
cd pipeline
python surveillance_pipeline.py
```
**Output:**
```
Initializing AegisVision pipeline...
All AegisVision modules loaded successfully
Surveillance pipeline ready

[Events continuously generated and logged]
```

### Terminal 3 - Detection API (Optional - for dual detection)
```bash
cd detection
python -m uvicorn api:app --reload --port 8001
```
**Output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8001
```

### Browser - Open Dashboard
```
http://127.0.0.1:8000/
```

**Complete System Output:**
```
┌─────────────────────────────────────────────────────────┐
│              BACKEND API RUNNING                        │
│         http://127.0.0.1:8000                           │
│                                                         │
│  ✓ Event Storage                                        │
│  ✓ Alert Management                                     │
│  ✓ WebSocket Streaming                                 │
│  ✓ Real-time Dashboard                                 │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│        SURVEILLANCE PIPELINE RUNNING                    │
│         (Detection + Analytics)                         │
│                                                         │
│  ✓ Video Input: Webcam                                  │
│  ✓ Face Detection: Enabled                              │
│  ✓ Movement Detection: Enabled                          │
│  ✓ Night Detection: Enabled                             │
│  ✓ Tracking: Enabled                                    │
│  ✓ Event Generation: Enabled                            │
└─────────────────────────────────────────────────────────┘

🎥 Frame 1234 Processing:
  📍 Detections: 3 objects (2 persons, 1 car)
  👁 Faces: 2 detected
  🚶 Movement: YES (15.3%)
  🌙 Night Mode: YES (brightness: 32)
  🔍 Track IDs: [42, 15, 8]
  
🚨 EVENT GENERATED:
  Type: night_movement
  Severity: HIGH
  Person: 42
  Timestamp: 2026-08-31T12:30:45

📤 Event sent to Backend API
🔔 Alert broadcasted to frontend via WebSocket
```

**Dashboard Display:**
```
Header: 🛡 AegisVision - 🟢 SYSTEM ACTIVE

Stats:
  Total Intrusion Alerts: 23
  System Status: ACTIVE
  AI Detection: ENABLED

Recent Alerts:
  🚨 INTRUSION: PERSON ID:42 (12:30:45)
     Zone: restricted_area_3
     Evidence: [Image Preview]
  
  🚨 NIGHT MOVEMENT (12:30:40)
     2 people detected
     Brightness: 32
  
  ⚠️ LOITERING: PERSON ID:15 (12:30:35)
     Duration: 45 seconds

WebSocket Connection: ✓ Connected
Alert Stream: ✓ Live
```

---

## 🔧 INTEGRATED PYTHON CODE EXAMPLE

### Create `run_complete_system.py`

```python
import subprocess
import sys
import os
import time
import threading

def run_backend():
    """Run backend API server"""
    print("[BACKEND] Starting FastAPI server...")
    os.chdir("backend")
    subprocess.run([
        sys.executable, "-m", "uvicorn", 
        "app.main:app", "--reload", "--port", "8000"
    ])

def run_pipeline():
    """Run main surveillance pipeline"""
    print("[PIPELINE] Starting surveillance pipeline...")
    time.sleep(2)  # Wait for backend to start
    os.chdir("../pipeline")
    subprocess.run([sys.executable, "surveillance_pipeline.py"])

def run_detection_api():
    """Run detection API (optional)"""
    print("[DETECTION] Starting detection API...")
    time.sleep(2)
    os.chdir("../detection")
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "api:app", "--reload", "--port", "8001"
    ])

def main():
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║     🛡 AegisVision - Complete Integrated System        ║
    ╚════════════════════════════════════════════════════════╝
    """)
    
    # Start all services in parallel
    services = [
        ("Backend API", run_backend),
        ("Surveillance Pipeline", run_pipeline),
        ("Detection API", run_detection_api),
    ]
    
    threads = []
    for name, func in services:
        thread = threading.Thread(target=func, daemon=True)
        thread.start()
        threads.append(thread)
    
    print("\n✓ All services started!")
    print("✓ Backend API: http://127.0.0.1:8000")
    print("✓ Dashboard: http://127.0.0.1:8000/")
    print("✓ Detection API: http://127.0.0.1:8001")
    print("✓ API Docs: http://127.0.0.1:8000/docs")
    
    # Keep running
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()
```

**Run it:**
```bash
cd d:\AegisVision
python run_complete_system.py
```

---

## 📊 DATA FLOW IN INTEGRATED SYSTEM

```
Frame 1 (Time: 12:30:45.001)
  ↓
Detection (YOLO11n)
  ├─ Person detected: ID=42, confidence=0.95, bbox=(100,50,200,300)
  ├─ Car detected: ID=8, confidence=0.92, bbox=(400,100,550,300)
  └─ Output: [detection_dict_1, detection_dict_2]
  ↓
Analytics
  ├─ Face Detection: Found 1 face
  ├─ Movement Detection: YES (15.3%)
  ├─ Night Detection: YES (brightness=32)
  └─ Output: {faces: [...], movement: True, low_light: True}
  ↓
Tracking-Intrusion Pipeline
  ├─ Track Person 42: In restricted zone
  ├─ Track Car 8: Outside zone
  └─ Output: [intrusion_event_person_42]
  ↓
Event Generator
  ├─ Combines analytics + tracking
  ├─ Creates event: {event_type: "intrusion", severity: "HIGH", ...}
  └─ Output: event_object
  ↓
Alert Manager
  ├─ Creates alert from event
  ├─ Sets severity level: 8/10
  └─ Output: alert_object
  ↓
Evidence Manager
  ├─ Saves frame to: evidence_data/evidence_1234.json
  └─ Output: file_path
  ↓
Backend API (POST /events)
  ├─ Receives event object
  ├─ Stores in database
  └─ Broadcasts via WebSocket
  ↓
Frontend Dashboard (WebSocket /ws/alerts)
  ├─ Receives alert notification
  ├─ Updates alert list
  ├─ Displays evidence image
  └─ Shows: "🚨 INTRUSION: PERSON ID:42 (12:30:45)"
```

---

## 🔍 MONITORING & DEBUGGING

### Check Backend Health
```bash
curl http://127.0.0.1:8000/health
```

### View All Events
```bash
curl http://127.0.0.1:8000/events | python -m json.tool
```

### View High-Severity Alerts
```bash
curl http://127.0.0.1:8000/alerts | python -m json.tool
```

### Stream Alerts via WebSocket
```python
import asyncio
import websockets
import json

async def stream_alerts():
    uri = "ws://127.0.0.1:8000/ws/alerts"
    async with websockets.connect(uri) as websocket:
        while True:
            alert = await websocket.recv()
            data = json.loads(alert)
            print(f"🚨 ALERT: {data['event_type']} - {data['severity']}")
            print(f"   Details: {data}")

asyncio.run(stream_alerts())
```

### Check Evidence Files
```bash
ls -la evidence_data/
cat evidence_data/evidence_1234.json | python -m json.tool
```

---

## 🎯 QUICK START COMMANDS

### Start Everything (Recommended)
```bash
cd d:\AegisVision
python run_complete_system.py
# Then open: http://127.0.0.1:8000
```

### Start Just Detection
```bash
cd d:\AegisVision\detection
python main.py
```

### Start Just Backend
```bash
cd d:\AegisVision\backend
python -m uvicorn app.main:app --reload
```

### Start Just Pipeline
```bash
cd d:\AegisVision\pipeline
python surveillance_pipeline.py
```

---

## ✅ SYSTEM STATUS CHECKLIST

- [ ] Backend API running (http://127.0.0.1:8000)
- [ ] Surveillance pipeline started
- [ ] Video input (webcam/file) active
- [ ] Face detection working
- [ ] Movement detection working
- [ ] Night detection working
- [ ] Tracking active
- [ ] Events generating
- [ ] Alerts sending
- [ ] Dashboard displaying
- [ ] WebSocket streaming
- [ ] Evidence saved

---

Generated: 2026-08-31
Complete Integration Guide & Runbook
