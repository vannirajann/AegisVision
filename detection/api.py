from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
import cv2
import os
import threading
import time
from datetime import datetime
import requests

app = FastAPI(title="AegisVision Detection")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "videos", "test.mp4")
ALERTS_DIR = os.path.join(BASE_DIR, "alerts")

os.makedirs(ALERTS_DIR, exist_ok=True)

print("Loading YOLO...")
model = YOLO(os.path.join(BASE_DIR, "yolo11n.pt"))
print("YOLO loaded successfully")

events = []
events_lock = threading.Lock()

latest_frame = None
frame_lock = threading.Lock()

current_intruders = 0

previous_positions = {}
counted_objects = set()


def create_intrusion_alert(frame, class_name, track_id, confidence):
    timestamp = datetime.now()

    filename = (
        f"intruder_{class_name}_"
        f"ID_{track_id}_"
        f"{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    )

    filepath = os.path.join(ALERTS_DIR, filename)
    cv2.imwrite(filepath, frame)

    event = {
        "event_type": "intrusion",
        "object": class_name,
        "track_id": int(track_id),
        "confidence": round(float(confidence), 2),
        "timestamp": timestamp.isoformat(),
        "evidence_file": filename
    }

    with events_lock:
        events.append(event)

        if len(events) > 100:
            events.pop(0)

    print("INTRUSION ALERT:", event)

    # Send to backend
    try:
        backend_event = {
            "event_type": "intrusion",
            "timestamp": timestamp.isoformat(),
            "source": "camera_01",
            "severity": "high",
            "data": {
                "object": class_name,
                "track_id": int(track_id),
                "confidence": round(float(confidence), 2),
                "evidence_file": filename
            }
        }

        response = requests.post(
            "http://127.0.0.1:8001/events",
            json=backend_event,
            timeout=3
        )

        if response.status_code == 200:
            print("BACKEND EVENT SENT SUCCESSFULLY")
        else:
            print("BACKEND ERROR:", response.status_code)

    except Exception as e:
        print("BACKEND CONNECTION ERROR:", e)


def detection_loop():

    global latest_frame
    global current_intruders

    print("Opening video:", VIDEO_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("ERROR: Cannot open video")
        return

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Video finished. Restarting...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        height, width = frame.shape[:2]

        # Restricted zone = upper half
        zone_y = height // 2

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        intruders = 0

        if results and results[0].boxes is not None:

            boxes = results[0].boxes

            for box in boxes:

                if box.id is None:
                    continue

                track_id = int(box.id[0])

                cls = int(box.cls[0])
                confidence = float(box.conf[0])

                class_name = model.names[cls]

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Draw detection
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"{class_name} ID:{track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                # Check line crossing
                if track_id in previous_positions:

                    previous_y = previous_positions[track_id]

                    crossed = (
                        previous_y > zone_y
                        and center_y <= zone_y
                    )

                    if crossed and track_id not in counted_objects:

                        counted_objects.add(track_id)
                        intruders += 1

                        create_intrusion_alert(
                            frame,
                            class_name,
                            track_id,
                            confidence
                        )

                previous_positions[track_id] = center_y

        current_intruders = intruders

        # Draw restricted line
        cv2.line(
            frame,
            (0, zone_y),
            (width, zone_y),
            (0, 0, 255),
            3
        )

        cv2.putText(
            frame,
            "RESTRICTED ZONE",
            (20, zone_y - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        with frame_lock:
            latest_frame = frame.copy()

        time.sleep(0.01)

    cap.release()


def generate_frames():

    while True:

        with frame_lock:

            if latest_frame is None:
                time.sleep(0.1)
                continue

            frame = latest_frame.copy()

        ret, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )


@app.get("/")
def home():
    return {
        "service": "AegisVision Detection",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "running",
        "service": "detection"
    }


@app.get("/events")
def get_events():

    with events_lock:
        return {
            "count": len(events),
            "events": list(events)
        }


@app.get("/status")
def status():

    return {
        "status": "running",
        "intruders": current_intruders,
        "events": len(events),
        "source": {
            "source_type": "video",
            "source": VIDEO_PATH
        }
    }


@app.get("/video-feed")
def video_feed():

    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


print("Starting detection thread...")

thread = threading.Thread(
    target=detection_loop,
    daemon=True
)

thread.start()