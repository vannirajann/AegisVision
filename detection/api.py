from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import cv2
import os
from datetime import datetime
import threading
import time

app = FastAPI()

# ==========================================
# FOLDERS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ALERTS_DIR = os.path.join(BASE_DIR, "alerts")
STATIC_DIR = os.path.join(BASE_DIR, "static")
VIDEO_PATH = os.path.join(BASE_DIR, "videos", "test.mp4")

os.makedirs(ALERTS_DIR, exist_ok=True)

# ==========================================
# YOLO MODEL
# ==========================================

model = YOLO("yolo11n.pt")

# person, bicycle, car, motorcycle, bus, truck
ALLOWED_CLASSES = [0, 1, 2, 3, 5, 7]

# ==========================================
# GLOBAL VARIABLES
# ==========================================

latest_frame = None
frame_lock = threading.Lock()

# Previous position of each tracked object
previous_positions = {}

# Objects already counted
counted_objects = set()

# ==========================================
# STATIC FILES
# ==========================================

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)

# ==========================================
# HOME DASHBOARD
# ==========================================

@app.get("/")
def home():

    return FileResponse(
        os.path.join(STATIC_DIR, "index.html")
    )


# ==========================================
# GET ALERTS
# ==========================================

@app.get("/alerts")
def get_alerts():

    files = []

    if os.path.exists(ALERTS_DIR):

        for file in os.listdir(ALERTS_DIR):

            if file.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                files.append(file)

    return {
        "total_alerts": len(files),
        "alerts": sorted(files, reverse=True)
    }


# ==========================================
# SHOW ALERT IMAGE
# ==========================================

@app.get("/alert-images/{filename}")
def get_alert_image(filename: str):

    image_path = os.path.join(
        ALERTS_DIR,
        filename
    )

    if os.path.exists(image_path):

        return FileResponse(image_path)

    return {
        "error": "Image not found"
    }


# ==========================================
# DETECTION LOOP
# ==========================================

def detection_loop():

    global latest_frame

    video = cv2.VideoCapture(VIDEO_PATH)

    while True:

        success, frame = video.read()

        # Restart video when finished
        if not success:

            video.set(
                cv2.CAP_PROP_POS_FRAMES,
                0
            )

            # IMPORTANT:
            # Do NOT clear counted_objects.
            # This prevents the same test video
            # from creating alerts again.

            previous_positions.clear()

            continue


        # Flip video
        frame = cv2.flip(frame, 1)

        height, width = frame.shape[:2]

        # Center crossing line
        ZONE_Y = height // 2

        # ==========================================
        # YOLO DETECTION + TRACKING
        # ==========================================

        results = model.track(
            frame,
            persist=True,
            classes=ALLOWED_CLASSES,
            tracker="bytetrack.yaml",
            verbose=False
        )

        output_frame = frame.copy()

        # ==========================================
        # DRAW RESTRICTED ZONE
        # ==========================================

        cv2.rectangle(
            output_frame,
            (0, 0),
            (width, ZONE_Y),
            (0, 0, 255),
            2
        )

        cv2.line(
            output_frame,
            (0, ZONE_Y),
            (width, ZONE_Y),
            (0, 0, 255),
            5
        )

        cv2.putText(
            output_frame,
            "RESTRICTED ZONE",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

        # ==========================================
        # CHECK DETECTED OBJECTS
        # ==========================================

        for box in results[0].boxes:

            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            x1, y1, x2, y2 = box.xyxy[0]

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            # Object center
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Get tracking ID
            if box.id is None:
                continue

            track_id = int(box.id[0])

            object_key = (
                f"{class_name}_{track_id}"
            )

            # Default
            color = (0, 255, 0)

            label = (
                f"{class_name.upper()} "
                f"ID:{track_id}"
            )

            # Previous position
            previous_y = previous_positions.get(
                object_key
            )

            # Check real crossing
            crossed_line = False

            if previous_y is not None:

                # Object moves from bottom
                # to top through red line
                if (
                    previous_y > ZONE_Y
                    and center_y <= ZONE_Y
                ):

                    crossed_line = True

            # Save current position
            previous_positions[object_key] = center_y

            # ==========================================
            # RESTRICTED AREA
            # ==========================================

            if center_y < ZONE_Y:

                color = (0, 0, 255)

                label = (
                    f"INTRUDER: "
                    f"{class_name.upper()}"
                )

            # ==========================================
            # CREATE ONE ALERT
            # ==========================================

            if crossed_line:

                # Use approximate crossing position
                # so ID changes do not create many alerts
                position_key = (
                    f"{class_name}_"
                    f"{center_x // 150}"
                )

                # Count only once
                if position_key not in counted_objects:

                    counted_objects.add(
                        position_key
                    )

                    timestamp = datetime.now().strftime(
                        "%Y%m%d_%H%M%S_%f"
                    )

                    filename = (
                        f"intruder_"
                        f"{class_name}_"
                        f"{timestamp}.jpg"
                    )

                    file_path = os.path.join(
                        ALERTS_DIR,
                        filename
                    )

                    # Draw box before saving
                    cv2.rectangle(
                        output_frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        3
                    )

                    cv2.putText(
                        output_frame,
                        f"ALERT: {class_name.upper()}",
                        (x1, max(y1 - 10, 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

                    # Save ONE evidence image
                    cv2.imwrite(
                        file_path,
                        output_frame
                    )

                    print(
                        f"ALERT +1: "
                        f"{class_name.upper()}"
                    )

            # ==========================================
            # DRAW OBJECT
            # ==========================================

            cv2.rectangle(
                output_frame,
                (x1, y1),
                (x2, y2),
                color,
                3
            )

            cv2.putText(
                output_frame,
                label,
                (x1, max(y1 - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        # Store latest frame
        with frame_lock:

            latest_frame = output_frame.copy()


# ==========================================
# START DETECTION
# ==========================================

@app.on_event("startup")
def start_detection():

    thread = threading.Thread(
        target=detection_loop,
        daemon=True
    )

    thread.start()


# ==========================================
# VIDEO STREAM
# ==========================================

def generate_frames():

    while True:

        with frame_lock:

            if latest_frame is None:

                frame = None

            else:

                frame = latest_frame.copy()

        if frame is None:

            time.sleep(0.05)
            continue

        _, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )

        time.sleep(0.03)


# ==========================================
# VIDEO FEED
# ==========================================

@app.get("/video-feed")
def video_feed():

    return StreamingResponse(
        generate_frames(),
        media_type=
        "multipart/x-mixed-replace; boundary=frame"
    )