# AegisVision - Complete Output Reference

## 🎬 DETECTION MODULE OUTPUT

### Main.py Output (Console)
```
🚨 INTRUSION DETECTED!
Object: PERSON
ID: 42
Evidence: alerts/intruder_person_ID_42_20260831_123045_123456.jpg
```

### Main.py Output (Files)
```
alerts/intruder_person_ID_42_20260831_123045_123456.jpg
alerts/intruder_car_ID_15_20260831_123050_234567.jpg
alerts/intruder_bicycle_ID_8_20260831_123055_345678.jpg
```

### API.py Output - GET /alerts
```json
{
  "total_alerts": 15,
  "alerts": [
    "intruder_person_ID_42_20260831_123045_123456.jpg",
    "intruder_car_ID_15_20260831_123050_234567.jpg",
    "intruder_bicycle_ID_8_20260831_123055_345678.jpg"
  ]
}
```

### Detection Results (Internal Data Structure)
```python
{
    "class_id": 0,
    "class_name": "person",
    "track_id": 42,
    "confidence": 0.95,
    "bbox": {
        "x1": 100,
        "y1": 50,
        "x2": 200,
        "y2": 300
    },
    "center": {
        "x": 150,
        "y": 175
    },
    "is_intruder": True,
    "timestamp": "2026-08-31T12:30:45.123456",
    "alert_image": "alerts/intruder_person_ID_42_20260831_123045_123456.jpg"
}
```

---

## 📊 ANALYTICS MODULE OUTPUT

### Face Detection Output
```python
{
    "faces_detected": 2,
    "faces": [
        {
            "id": 1,
            "confidence": 0.98,
            "bbox": {
                "x": 150,
                "y": 80,
                "width": 60,
                "height": 80
            },
            "area": 4800
        },
        {
            "id": 2,
            "confidence": 0.92,
            "bbox": {
                "x": 400,
                "y": 120,
                "width": 55,
                "height": 75
            },
            "area": 4125
        }
    ]
}
```

### Movement Detection Output
```python
{
    "movement_detected": True,
    "movement_percentage": 15.3,
    "motion_region": {
        "x": 100,
        "y": 50,
        "width": 300,
        "height": 250
    },
    "intensity": "HIGH",
    "changed_pixels": 12453
}
```

### Night/Low-Light Detection Output
```python
{
    "low_light": True,
    "brightness": 32,
    "contrast": 15,
    "status": "night",
    "infrared_required": True,
    "confidence": 0.94,
    "light_level": {
        "min": 10,
        "max": 85,
        "avg": 32
    }
}
```

### Loitering Detection Output
```python
{
    "loitering_detected": True,
    "loitering_objects": [
        {
            "track_id": 42,
            "class": "person",
            "stationary_time": 45.2,  # seconds
            "location": {"x": 200, "y": 300},
            "threshold_time": 30,
            "alert_level": "HIGH",
            "description": "Person stationary for 45.2 seconds (threshold: 30s)"
        }
    ]
}
```

### Complete Analytics Frame Output
```python
{
    "timestamp": "2026-08-31T12:30:45.123456",
    "frame_id": 1234,
    "faces": {...},
    "people": [
        {
            "id": 42,
            "class": "person",
            "confidence": 0.95,
            "bbox": {...}
        }
    ],
    "movement": {
        "detected": True,
        "percentage": 15.3
    },
    "low_light": True,
    "brightness": 32,
    "light_status": "night",
    "events": [
        {
            "type": "night_movement",
            "severity": "high",
            "person_id": 42
        }
    ]
}
```

---

## 🔄 TRACKING-INTRUSION PIPELINE OUTPUT

### Centroid Tracking Output
```python
{
    "track_id": 42,
    "centroid": (150, 175),
    "bbox": (100, 50, 200, 300),
    "class": "person",
    "confidence": 0.95,
    "frame_count": 15,
    "age": 15,
    "status": "active"
}
```

### Virtual Line Crossing Output
```python
{
    "crossing_detected": True,
    "track_id": 42,
    "class": "person",
    "crossing_point": (200, 240),
    "timestamp": "2026-08-31T12:30:45",
    "direction": "upward",
    "confidence": 0.95,
    "event": {
        "event_type": "intrusion",
        "track_id": 42,
        "object_type": "person",
        "confidence": 0.95,
        "crossing_details": {
            "line": [(0, 240), (640, 240)],
            "point": (200, 240),
            "direction": "upward"
        }
    }
}
```

### Restricted Zone Entry Output
```python
{
    "zone_entry_detected": True,
    "track_id": 42,
    "class": "person",
    "entry_point": (150, 175),
    "zone_id": "restricted_zone_1",
    "timestamp": "2026-08-31T12:30:45",
    "zone_definition": {
        "type": "polygon",
        "points": [(100, 100), (300, 100), (300, 400), (100, 400)]
    },
    "event": {
        "event_type": "intrusion",
        "track_id": 42,
        "object_type": "person",
        "confidence": 0.95,
        "zone_details": {
            "zone_name": "restricted_zone_1",
            "entry_point": (150, 175),
            "zone_area": 90000
        }
    }
}
```

### Complete Pipeline Frame Output
```python
[
    {
        "event_type": "intrusion",
        "track_id": 42,
        "object_type": "person",
        "confidence": 0.95,
        "event_subtype": "zone_entry",
        "zone_name": "restricted_area_3",
        "timestamp": "2026-08-31T12:30:45.123456",
        "frame_number": 1234,
        "detection_data": {
            "bbox": (100, 50, 200, 300),
            "centroid": (150, 175),
            "class_id": 0
        }
    }
]
```

---

## 📋 EVENT GENERATOR OUTPUT

### Generated Event Object
```python
{
    "event_id": "EVT_20260831_123045_001",
    "event_type": "night_movement",
    "timestamp": "2026-08-31T12:30:45.123456",
    "source_module": "analytics",
    "severity": "HIGH",
    "person_id": 42,
    "details": {
        "message": "Night-time movement detected",
        "people_count": 1,
        "brightness": 32,
        "light_status": "night",
        "confidence": 0.95
    },
    "metadata": {
        "frame_id": 1234,
        "camera_id": "camera_01",
        "zone": "zone_3"
    }
}
```

### Multiple Events in Single Frame
```python
[
    {
        "event_id": "EVT_20260831_123045_001",
        "event_type": "night_movement",
        "severity": "HIGH",
        "person_id": 42
    },
    {
        "event_id": "EVT_20260831_123045_002",
        "event_type": "intrusion",
        "severity": "CRITICAL",
        "person_id": 42,
        "zone": "restricted_area_3"
    },
    {
        "event_id": "EVT_20260831_123045_003",
        "event_type": "loitering",
        "severity": "MEDIUM",
        "person_id": 15
    }
]
```

---

## 🔔 ALERT MANAGER OUTPUT

### Alert Object
```python
{
    "alert_id": "ALT_20260831_123045_001",
    "event_id": "EVT_20260831_123045_001",
    "alert_type": "night_movement",
    "severity": "HIGH",
    "level": 8,  # 1-10 scale
    "status": "active",
    "timestamp": "2026-08-31T12:30:45.123456",
    "person_id": 42,
    "actions": [
        "NOTIFY_ADMIN",
        "LOG_EVENT",
        "CAPTURE_EVIDENCE"
    ],
    "metadata": {
        "camera": "camera_01",
        "zone": "zone_3",
        "confidence": 0.95
    }
}
```

---

## 💾 EVIDENCE MANAGER OUTPUT

### Evidence File Path
```
evidence_data/evidence_1234.json
evidence_data/evidence_1235.json
evidence_data/evidence_1236.json
```

### Evidence JSON File Contents
```json
{
    "evidence_id": "EVT_20260831_123045_001",
    "timestamp": "2026-08-31T12:30:45.123456",
    "event_type": "night_movement",
    "severity": "HIGH",
    "camera_id": "camera_01",
    "person_id": 42,
    "details": {
        "message": "Night-time movement detected",
        "people_count": 1,
        "brightness": 32,
        "light_status": "night",
        "alert_level": 8
    },
    "metadata": {
        "frame_id": 1234,
        "zone": "zone_3",
        "detection_confidence": 0.95
    },
    "frame_image": "path/to/frame_1234.jpg"
}
```

---

## 🌐 BACKEND API OUTPUT

### POST /events Response
```json
{
    "status": "success",
    "event_id": "EVT_20260831_123045_001",
    "message": "Event stored successfully",
    "timestamp": "2026-08-31T12:30:45.123456"
}
```

### GET /events Response
```json
{
    "total_events": 125,
    "events": [
        {
            "event_id": "EVT_20260831_123045_001",
            "event_type": "intrusion",
            "timestamp": "2026-08-31T12:30:45",
            "source": "camera_01",
            "severity": "high",
            "data": {
                "track_id": 42,
                "zone": "restricted_area_3",
                "object_type": "person",
                "confidence": 0.95
            }
        },
        {
            "event_id": "EVT_20260831_123040_002",
            "event_type": "loitering",
            "timestamp": "2026-08-31T12:30:40",
            "source": "camera_01",
            "severity": "medium",
            "data": {
                "person_id": 15,
                "zone": "zone_3",
                "stationary_time": 45.2
            }
        }
    ],
    "page": 1,
    "page_size": 50
}
```

### GET /alerts Response (High Severity Only)
```json
{
    "total_alerts": 23,
    "alerts": [
        {
            "alert_id": "ALT_20260831_123045_001",
            "event_type": "intrusion",
            "timestamp": "2026-08-31T12:30:45",
            "severity": "high",
            "data": {
                "track_id": 42,
                "object_type": "person",
                "zone": "restricted_area_3"
            }
        },
        {
            "alert_id": "ALT_20260831_123040_002",
            "event_type": "night_movement",
            "timestamp": "2026-08-31T12:30:40",
            "severity": "high",
            "data": {
                "brightness": 32,
                "people_count": 2
            }
        }
    ]
}
```

### WebSocket /ws/alerts Message
```json
{
    "alert_id": "ALT_20260831_123045_001",
    "event_type": "intrusion",
    "timestamp": "2026-08-31T12:30:45.123456",
    "source": "camera_01",
    "severity": "high",
    "message": "🚨 INTRUSION DETECTED in restricted_area_3!",
    "data": {
        "track_id": 42,
        "object_type": "person",
        "confidence": 0.95,
        "zone": "restricted_area_3",
        "camera": "camera_01"
    }
}
```

---

## 🎨 FRONTEND DASHBOARD OUTPUT

### Dashboard Display Elements
```html
<!-- Header -->
🛡 AegisVision - 🟢 SYSTEM ACTIVE

<!-- Statistics Cards -->
Total Intrusion Alerts: 23
System Status: ACTIVE
AI Detection: ENABLED

<!-- Main Dashboard -->
Live Video Feed: [Video Stream]

Recent Alerts:
├── 🚨 INTRUSION: PERSON ID:42 (12:30:45)
│   └── [Evidence Image Preview]
├── 🚨 NIGHT MOVEMENT (12:30:40)
│   └── 2 people detected in low-light
├── ⚠️ LOITERING: PERSON ID:15 (12:30:35)
│   └── Stationary for 45 seconds
└── 🚨 INTRUSION: CAR ID:8 (12:30:30)
    └── [Evidence Image Preview]

<!-- Alert Details Modal -->
Event Type: intrusion
Object: PERSON
Track ID: 42
Confidence: 95%
Timestamp: 2026-08-31 12:30:45
Zone: restricted_area_3
Evidence: [Full Image]
```

---

## 📈 MAIN PIPELINE OUTPUT

### Process Frame Return Value
```python
{
    "faces": [
        {
            "id": 1,
            "confidence": 0.98,
            "bbox": {...}
        }
    ],
    "people": [
        {
            "id": 42,
            "class": "person",
            "confidence": 0.95,
            "bbox": {...}
        }
    ],
    "movement": True,
    "low_light": True,
    "brightness": 32,
    "light_status": "night",
    "event": {
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
}
```

---

## 📝 SUMMARY TABLE

| Module | Output Type | Format | Destination |
|--------|------------|--------|-------------|
| Detection | Images | JPEG files | `alerts/` folder |
| Detection | Console | Text | Terminal output |
| Detection API | Alerts List | JSON | HTTP response |
| Analytics | Detection Results | Dict | Internal memory |
| Analytics | Events | List of dicts | Event generator |
| Tracking | Intrusion Events | Dicts | Backend |
| Events | Event Objects | JSON dicts | Alert manager |
| Alerts | Alert Objects | JSON dicts | Backend API |
| Evidence | JSON files | JSON | `evidence_data/` folder |
| Backend | API Response | JSON | HTTP/WebSocket |
| Frontend | Dashboard | HTML/CSS | Browser |

---

Generated: 2026-08-31
Complete Integration & Output Reference
