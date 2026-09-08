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

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_PATH = os.path.join(
    BASE_DIR,
    "videos",
    "test.mp4"
)

ALERTS_DIR = os.path.join(
    BASE_DIR,
    "alerts"
)

os.makedirs(ALERTS_DIR, exist_ok=True)


# --------------------------------------------------
# LOAD YOLO
# --------------------------------------------------

print("Loading YOLO...")

model = YOLO(
    os.path.join(
        BASE_DIR,
        "yolo11n.pt"
    )
)

print("YOLO loaded successfully")


# --------------------------------------------------
# GLOBAL VARIABLES
# --------------------------------------------------

events = []

events_lock = threading.Lock()

latest_frame = None

frame_lock = threading.Lock()

current_intruders = 0

# Store previous Y position of every tracked object
previous_positions = {}

# IMPORTANT:
# A track ID can generate only ONE alert
alerted_tracks = set()


# --------------------------------------------------
# CREATE INTRUSION ALERT
# --------------------------------------------------

def create_intrusion_alert(
    frame,
    class_name,
    track_id,
    confidence
):

    timestamp = datetime.now()

    filename = (
        f"intruder_{class_name}_"
        f"ID_{track_id}_"
        f"{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    )

    filepath = os.path.join(
        ALERTS_DIR,
        filename
    )

    # Save evidence image
    cv2.imwrite(
        filepath,
        frame
    )

    event = {
        "event_type": "intrusion",
        "object": class_name,
        "track_id": int(track_id),
        "confidence": round(
            float(confidence),
            2
        ),
        "timestamp": timestamp.isoformat(),
        "evidence_file": filename
    }

    # Store event
    with events_lock:

        events.append(event)

        if len(events) > 100:

            events.pop(0)

    print(
        "INTRUSION ALERT:",
        event
    )

    # --------------------------------------------------
    # SEND EVENT TO BACKEND
    # --------------------------------------------------

    try:

        backend_event = {

            "event_type": "intrusion",

            "timestamp": timestamp.isoformat(),

            "source": "camera_01",

            "severity": "high",

            "data": {

                "object": class_name,

                "track_id": int(track_id),

                "confidence": round(
                    float(confidence),
                    2
                ),

                "evidence_file": filename
            }
        }

        response = requests.post(

            "http://127.0.0.1:8001/events",

            json=backend_event,

            timeout=3
        )

        if response.status_code == 200:

            print(
                "BACKEND EVENT SENT SUCCESSFULLY"
            )

        else:

            print(
                "BACKEND ERROR:",
                response.status_code
            )

    except Exception as e:

        print(
            "BACKEND CONNECTION ERROR:",
            e
        )


# --------------------------------------------------
# DETECTION LOOP
# --------------------------------------------------

def detection_loop():

    global latest_frame

    global current_intruders

    print(
        "Opening video:",
        VIDEO_PATH
    )

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():

        print(
            "ERROR: Cannot open video"
        )

        return

    while True:

        ret, frame = cap.read()

        # --------------------------------------------------
        # VIDEO RESTART
        # --------------------------------------------------

        if not ret:

            print(
                "Video finished. Restarting..."
            )

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                0
            )

            # Reset tracking information
            previous_positions.clear()

            # Reset alert IDs for the new video run
            alerted_tracks.clear()

            continue


        # --------------------------------------------------
        # FRAME INFORMATION
        # --------------------------------------------------

        height, width = frame.shape[:2]

        # Restricted line = middle of video
        zone_y = height // 2


        # --------------------------------------------------
        # YOLO + BYTE TRACK
        # --------------------------------------------------

        results = model.track(

            frame,

            persist=True,

            tracker="bytetrack.yaml",

            verbose=False
        )

        intruders = 0

        current_track_ids = set()


        # --------------------------------------------------
        # PROCESS DETECTIONS
        # --------------------------------------------------

        if (
            results
            and results[0].boxes is not None
        ):

            boxes = results[0].boxes

            for box in boxes:

                # Skip objects without tracking ID
                if box.id is None:

                    continue


                # ------------------------------------------
                # TRACK ID
                # ------------------------------------------

                track_id = int(
                    box.id[0]
                )

                current_track_ids.add(
                    track_id
                )


                # ------------------------------------------
                # CLASS
                # ------------------------------------------

                cls = int(
                    box.cls[0]
                )

                class_name = model.names[
                    cls
                ]


                # ------------------------------------------
                # CONFIDENCE
                # ------------------------------------------

                confidence = float(
                    box.conf[0]
                )


                # ------------------------------------------
                # BOUNDING BOX
                # ------------------------------------------

                x1, y1, x2, y2 = map(

                    int,

                    box.xyxy[0]
                )


                # ------------------------------------------
                # CENTER
                # ------------------------------------------

                center_x = (
                    x1 + x2
                ) // 2

                center_y = (
                    y1 + y2
                ) // 2


                # ------------------------------------------
                # DRAW BOX
                # ------------------------------------------

                cv2.rectangle(

                    frame,

                    (x1, y1),

                    (x2, y2),

                    (0, 255, 0),

                    2
                )


                # ------------------------------------------
                # DRAW LABEL
                # ------------------------------------------

                cv2.putText(

                    frame,

                    f"{class_name} "
                    f"ID:{track_id} "
                    f"{confidence:.2f}",

                    (
                        x1,
                        max(
                            y1 - 10,
                            20
                        )
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.6,

                    (0, 255, 0),

                    2
                )


                # --------------------------------------------------
                # ONLY PERSON CAN TRIGGER INTRUSION
                # --------------------------------------------------

                if class_name != "person":

                    previous_positions[
                        track_id
                    ] = center_y

                    continue


                # --------------------------------------------------
                # CHECK CROSSING
                # --------------------------------------------------

                if track_id in previous_positions:

                    previous_y = (
                        previous_positions[
                            track_id
                        ]
                    )


                    # Person moves from BELOW
                    # restricted line
                    #
                    # to ABOVE restricted line
                    crossed_into_zone = (

                        previous_y > zone_y

                        and

                        center_y <= zone_y
                    )


                    # --------------------------------------------------
                    # INTRUSION CONDITION
                    # --------------------------------------------------

                    if crossed_into_zone:

                        # Person must have confidence >= 0.50
                        if confidence >= 0.50:

                            # ------------------------------------------
                            # IMPORTANT DUPLICATE PREVENTION
                            # ------------------------------------------

                            if (
                                track_id
                                not in alerted_tracks
                            ):

                                # Immediately mark this ID
                                # as alerted
                                alerted_tracks.add(
                                    track_id
                                )

                                intruders += 1


                                # Create alert
                                create_intrusion_alert(

                                    frame,

                                    class_name,

                                    track_id,

                                    confidence
                                )


                # Update previous position
                previous_positions[
                    track_id
                ] = center_y


        # --------------------------------------------------
        # REMOVE OLD TRACK IDs
        # --------------------------------------------------

        old_tracks = (

            set(
                previous_positions.keys()
            )

            -

            current_track_ids
        )


        for old_track in old_tracks:

            previous_positions.pop(
                old_track,
                None
            )


        # --------------------------------------------------
        # CURRENT INTRUSION COUNT
        # --------------------------------------------------

        current_intruders = intruders


        # --------------------------------------------------
        # DRAW RESTRICTED LINE
        # --------------------------------------------------

        cv2.line(

            frame,

            (0, zone_y),

            (width, zone_y),

            (0, 0, 255),

            3
        )


        # --------------------------------------------------
        # RESTRICTED ZONE LABEL
        # --------------------------------------------------

        cv2.putText(

            frame,

            "RESTRICTED ZONE",

            (
                20,
                max(
                    zone_y - 15,
                    30
                )
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 0, 255),

            2
        )


        # --------------------------------------------------
        # INTRUSION COUNT
        # --------------------------------------------------

        cv2.putText(

            frame,

            f"New Intrusions: "
            f"{current_intruders}",

            (20, 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 0, 255),

            2
        )


        # --------------------------------------------------
        # SAVE LATEST FRAME
        # --------------------------------------------------

        with frame_lock:

            latest_frame = frame.copy()


        time.sleep(0.01)


    cap.release()


# --------------------------------------------------
# VIDEO STREAM
# --------------------------------------------------

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

            b"Content-Type: "
            b"image/jpeg\r\n\r\n"

            + buffer.tobytes()

            + b"\r\n"
        )


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.get("/")
def home():

    return {

        "service":
        "AegisVision Detection",

        "status":
        "running"
    }


# --------------------------------------------------
# HEALTH
# --------------------------------------------------

@app.get("/health")
def health():

    return {

        "status":
        "running",

        "service":
        "detection"
    }


# --------------------------------------------------
# EVENTS
# --------------------------------------------------

@app.get("/events")
def get_events():

    with events_lock:

        return {

            "count":
            len(events),

            "events":
            list(events)
        }


# --------------------------------------------------
# STATUS
# --------------------------------------------------

@app.get("/status")
def status():

    return {

        "status":
        "running",

        "intruders":
        current_intruders,

        "events":
        len(events),

        "source": {

            "source_type":
            "video",

            "source":
            VIDEO_PATH
        }
    }


# --------------------------------------------------
# VIDEO FEED
# --------------------------------------------------

@app.get("/video-feed")
def video_feed():

    return StreamingResponse(

        generate_frames(),

        media_type=
        "multipart/x-mixed-replace; "
        "boundary=frame"
    )


# --------------------------------------------------
# START DETECTION THREAD
# --------------------------------------------------

print(
    "Starting detection thread..."
)

thread = threading.Thread(

    target=detection_loop,

    daemon=True
)

thread.start()


# --------------------------------------------------
# START SERVER
# --------------------------------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        app,

        host="127.0.0.1",

        port=8000
    )