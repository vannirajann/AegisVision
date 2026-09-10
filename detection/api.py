from fastapi import FastAPI, Body
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ultralytics import YOLO

import cv2
import os
import threading
import time

from datetime import datetime


# ==========================================
# FASTAPI APP
# ==========================================

app = FastAPI()


# ==========================================
# PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STATIC_DIR = os.path.join(BASE_DIR, "static")

VIDEOS_DIR = os.path.join(BASE_DIR, "videos")

ALERTS_DIR = os.path.join(BASE_DIR, "alerts")

MODELS_DIR = os.path.join(BASE_DIR, "models")

VIDEO_PATH = os.path.join(
    VIDEOS_DIR,
    "test.mp4"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "yolo11n.pt"
)

FACE_CASCADE_PATH = os.path.join(
    MODELS_DIR,
    "haarcascade_frontalface_default.xml"
)


# ==========================================
# CREATE FOLDERS
# ==========================================

os.makedirs(STATIC_DIR, exist_ok=True)

os.makedirs(VIDEOS_DIR, exist_ok=True)

os.makedirs(ALERTS_DIR, exist_ok=True)

os.makedirs(MODELS_DIR, exist_ok=True)


# ==========================================
# LOAD YOLO
# ==========================================

print("Loading YOLO...")

model = YOLO(MODEL_PATH)

print("YOLO loaded successfully")


# ==========================================
# LOAD FACE DETECTOR
# ==========================================

face_cascade = None

if os.path.exists(FACE_CASCADE_PATH):

    cascade = cv2.CascadeClassifier(
        FACE_CASCADE_PATH
    )

    if not cascade.empty():

        face_cascade = cascade

        print(
            "Face detection module loaded successfully"
        )

    else:

        print(
            "WARNING: Invalid face cascade"
        )

else:

    print(
        "WARNING: Face cascade not found"
    )


# ==========================================
# ALLOWED YOLO CLASSES
# ==========================================

ALLOWED_CLASSES = [

    0,   # person
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    5,   # bus
    7    # truck

]


# ==========================================
# GLOBAL VARIABLES
# ==========================================

latest_frame = None

frame_lock = threading.Lock()

# Current number of intruders in the latest frame
current_intruders = 0


events = []

events_lock = threading.Lock()


previous_positions = {}

counted_objects = set()


source_lock = threading.Lock()


current_source_type = "video"

current_source = VIDEO_PATH

source_changed = False


# ==========================================
# STATIC FILES
# ==========================================

app.mount(

    "/static",

    StaticFiles(
        directory=STATIC_DIR
    ),

    name="static"

)


# ==========================================
# HOME
# ==========================================

@app.get("/")
def home():

    return FileResponse(

        os.path.join(
            STATIC_DIR,
            "index.html"
        )

    )


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health():

    return {

        "status": "success",

        "message": "AegisVision is running"

    }


# ==========================================
# GET EVENTS
# ==========================================

@app.get("/events")
def get_events():

    with events_lock:

        return {

            "total_events": len(events),

            "events": list(
                reversed(events)
            )

        }


# ==========================================
# GET ALERTS
# ==========================================

@app.get("/alerts")
def get_alerts():

    alert_files = []


    for filename in os.listdir(ALERTS_DIR):

        if filename.lower().endswith(

            (
                ".jpg",
                ".jpeg",
                ".png"
            )

        ):

            alert_files.append(filename)


    return {

        "total_alerts": len(alert_files),

        "alerts": sorted(
            alert_files,
            reverse=True
        )

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

        return FileResponse(
            image_path
        )


    return {

        "error": "Image not found"

    }


# ==========================================
# GET CURRENT SOURCE
# ==========================================

@app.get("/source")
def get_source():

    with source_lock:

        return {

            "source_type":
            current_source_type,

            "source":
            str(current_source)

        }


# ==========================================
# SET SOURCE
# COMPATIBLE WITH YOUR HTML
# ==========================================

@app.post("/set-source")
def set_source(data: dict = Body(...)):

    global current_source_type
    global current_source
    global source_changed


    source = data.get("source")


    with source_lock:


        # ==============================
        # VIDEO
        # ==============================

        if source == "video":

            current_source_type = "video"

            current_source = VIDEO_PATH

            source_changed = True


            return {

                "status": "success",

                "source_name": "Video File"

            }


        # ==============================
        # WEBCAM
        # ==============================

        if source == "webcam":

            current_source_type = "webcam"

            current_source = 0

            source_changed = True


            return {

                "status": "success",

                "source_name": "Webcam"

            }


        # ==============================
        # RTSP
        # ==============================

        if source == "rtsp":

            rtsp_url = data.get(
                "rtsp_url"
            )


            if not rtsp_url:

                return {

                    "status": "error",

                    "message":
                    "RTSP URL is required"

                }


            current_source_type = "rtsp"

            current_source = rtsp_url

            source_changed = True


            return {

                "status": "success",

                "source_name":
                "RTSP / IP CCTV"

            }


    return {

        "status": "error",

        "message": "Invalid source"

    }


# ==========================================
# FACE DETECTION
# ==========================================

def detect_faces_in_person(

    frame,

    x1,
    y1,
    x2,
    y2

):

    faces_found = []


    if face_cascade is None:

        return faces_found


    height, width = frame.shape[:2]


    x1 = max(0, x1)

    y1 = max(0, y1)

    x2 = min(width, x2)

    y2 = min(height, y2)


    person_roi = frame[
        y1:y2,
        x1:x2
    ]


    if person_roi.size == 0:

        return faces_found


    gray = cv2.cvtColor(

        person_roi,

        cv2.COLOR_BGR2GRAY

    )


    detected_faces = face_cascade.detectMultiScale(

        gray,

        scaleFactor=1.1,

        minNeighbors=5,

        minSize=(30, 30)

    )


    for fx, fy, fw, fh in detected_faces:


        faces_found.append(

            (

                x1 + fx,

                y1 + fy,

                x1 + fx + fw,

                y1 + fy + fh

            )

        )


    return faces_found


# ==========================================
# SAVE FACE CROP
# ==========================================

def save_face_crop(

    frame,

    face,

    track_id,

    timestamp

):

    fx1, fy1, fx2, fy2 = face


    height, width = frame.shape[:2]


    padding = 40


    x1 = max(
        0,
        fx1 - padding
    )

    y1 = max(
        0,
        fy1 - padding
    )

    x2 = min(
        width,
        fx2 + padding
    )

    y2 = min(
        height,
        fy2 + padding
    )


    crop = frame[
        y1:y2,
        x1:x2
    ]


    if crop.size == 0:

        return None


    crop = cv2.resize(

        crop,

        None,

        fx=3,

        fy=3,

        interpolation=cv2.INTER_CUBIC

    )


    filename = (

        f"face_"

        f"person_ID_{track_id}_"

        f"{timestamp}.jpg"

    )


    file_path = os.path.join(

        ALERTS_DIR,

        filename

    )


    success = cv2.imwrite(

        file_path,

        crop

    )


    if success:

        print(

            "FACE IMAGE SAVED:",

            filename

        )

        return filename


    return None


# ==========================================
# CREATE INTRUSION ALERT
# ==========================================

def create_intrusion_alert(

    frame,

    class_name,

    track_id,

    confidence,

    faces

):

    now = datetime.now()


    timestamp = now.strftime(

        "%Y%m%d_%H%M%S_%f"

    )


    evidence_filename = (

        f"intruder_"

        f"{class_name}_"

        f"ID_{track_id}_"

        f"{timestamp}.jpg"

    )


    evidence_path = os.path.join(

        ALERTS_DIR,

        evidence_filename

    )


    saved = cv2.imwrite(

        evidence_path,

        frame

    )


    if saved:

        print(

            "EVIDENCE SAVED:",

            evidence_filename

        )


    face_filename = None


    # ======================================
    # SAVE LARGEST FACE
    # ======================================

    if (

        class_name == "person"

        and

        len(faces) > 0

    ):


        largest_face = max(

            faces,

            key=lambda f:

            (

                f[2] - f[0]

            )

            *

            (

                f[3] - f[1]

            )

        )


        face_filename = save_face_crop(

            frame,

            largest_face,

            track_id,

            timestamp

        )


    event = {

        "event_type":
        "intrusion",

        "object":
        class_name,

        "track_id":
        track_id,

        "confidence":
        round(
            confidence,
            2
        ),

        "timestamp":
        now.isoformat(),

        "evidence_file":
        evidence_filename,

        "face_file":
        face_filename

    }


    with events_lock:

        events.append(
            event
        )


    print(

        f"ALERT CREATED: "

        f"{class_name.upper()} "

        f"ID:{track_id}"

    )


# ==========================================
# DETECTION LOOP
# ==========================================

def detection_loop():

    global latest_frame

    video = None


    using_file = video.isOpened()

    if not using_file:
        print("Video file not found, using webcam...")
        video = cv2.VideoCapture(0)

    while True:


        try:


            # ==================================
            # GET SOURCE
            # ==================================

            with source_lock:

                selected_source = current_source

                selected_type = current_source_type

                changed = source_changed


            # ==================================
            # OPEN SOURCE
            # ==================================

            if video is None or changed:


                if video is not None:

                    video.release()


                print(

                    "Opening source:",

                    selected_type

                )


                video = cv2.VideoCapture(
                    selected_source
                )


                with source_lock:

                    source_changed = False


                previous_positions.clear()

                counted_objects.clear()


                time.sleep(0.5)


            # ==================================
            # SOURCE ERROR
            # ==================================

            if not video.isOpened():

                print(
                    "ERROR: Cannot open source"
                )

                time.sleep(1)

                continue


            # ==================================
            # READ FRAME
            # ==================================

            success, frame = video.read()

        # Restart video when finished
        if not success:

                    video.set(

                        cv2.CAP_PROP_POS_FRAMES,

                        0

                    )


                    previous_positions.clear()

                    counted_objects.clear()


                    continue


                else:

                    time.sleep(0.5)

                    continue


            # ==================================
            # RESIZE LARGE VIDEOS
            # ==================================

            height, width = frame.shape[:2]


            if width > 1280:


                scale = 1280 / width


                frame = cv2.resize(

                    frame,

                    None,

                    fx=scale,

                    fy=scale

                )


            height, width = frame.shape[:2]


            # ==================================
            # RESTRICTED ZONE
            # TOP HALF
            # ==================================

            ZONE_Y = height // 2


            # ==================================
            # YOLO TRACKING
            # ==================================

            results = model.track(

                frame,

                persist=True,

                classes=ALLOWED_CLASSES,

                tracker="bytetrack.yaml",

                conf=0.4,

                verbose=False

            )


            output_frame = frame.copy()


            # ==================================
            # DRAW RESTRICTED AREA
            # ==================================

            cv2.rectangle(

                output_frame,

                (0, 0),

                (
                    width,
                    ZONE_Y
                ),

                (
                    0,
                    0,
                    255
                ),

                3

            )


            cv2.putText(

                output_frame,

                "RESTRICTED ZONE",

                (
                    20,
                    45
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                1,

                (
                    0,
                    0,
                    255
                ),

                3

            )

        # ==========================================
        # CHECK DETECTED OBJECTS
        # ==========================================

                for box in results[0].boxes:


                    # ==========================
                    # CLASS
                    # ==========================

                    class_id = int(
                        box.cls[0]
                    )


                    class_name = model.names[
                        class_id
                    ]


                    # ==========================
                    # CONFIDENCE
                    # ==========================

                    confidence = float(
                        box.conf[0]
                    )


                    # ==========================
                    # BOX
                    # ==========================

                    x1, y1, x2, y2 = box.xyxy[0]


                    x1 = int(x1)

                    y1 = int(y1)

                    x2 = int(x2)

                    y2 = int(y2)


                    # ==========================
                    # TRACK ID
                    # ==========================

                    if box.id is None:

                        continue


                    track_id = int(
                        box.id[0]
                    )


                    object_key = (

                        f"{class_name}_"

                        f"{track_id}"

                    )


                    # ==========================
                    # CENTER
                    # ==========================

                    center_y = (

                        y1 + y2

                    ) // 2


                    previous_y = previous_positions.get(
                        object_key
                    )


                    previous_positions[
                        object_key
                    ] = center_y


                    # ==========================
                    # CHECK CROSSING
                    # ==========================

                    crossed_line = False


                    if previous_y is not None:


                        if (

                            previous_y > ZONE_Y

                            and

                            center_y <= ZONE_Y

                        ):

                            crossed_line = True


                    # ==========================
                    # COLOR
                    # ==========================

                    color = (

                color = (0, 0, 255)

                    )


                    label = (

                        f"{class_name.upper()} "

                        f"ID:{track_id}"

                    )


                    if center_y <= ZONE_Y:


                        color = (

                            0,
                            0,
                            255

                        )


                        label = (

                            f"INTRUDER "

                            f"{class_name.upper()} "

                            f"ID:{track_id}"

                        )


                    # ==========================
                    # FACE DETECTION
                    # ==========================

                    faces = []


                    if class_name == "person":


                        faces = detect_faces_in_person(

                            frame,

                            x1,
                            y1,
                            x2,
                            y2

                        )


                        for fx1, fy1, fx2, fy2 in faces:


                            cv2.rectangle(

                                output_frame,

                                (
                                    fx1,
                                    fy1
                                ),

                                (
                                    fx2,
                                    fy2
                                ),

                                (
                                    255,
                                    0,
                                    255
                                ),

                                2

                            )


                            cv2.putText(

                                output_frame,

                                "FACE DETECTED",

                                (
                                    fx1,

                                    max(
                                        fy1 - 10,
                                        20
                                    )

                                ),

                                cv2.FONT_HERSHEY_SIMPLEX,

                                0.5,

                                (
                                    255,
                                    0,
                                    255
                                ),

                                2

                            )


                    # ==========================
                    # DRAW OBJECT
                    # ==========================

                    cv2.rectangle(

                        output_frame,

                        (
                            x1,
                            y1
                        ),

                        (
                            x2,
                            y2
                        ),

                        color,

                        3

                    )


                    cv2.putText(

                        output_frame,

                        label,

                        (
                            x1,

                            max(
                                y1 - 10,
                                30
                            )

                        ),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.7,

                        color,

                        2

                    )

        # Store latest frame
        with frame_lock:

                latest_frame = output_frame.copy()


            time.sleep(0.01)


        except Exception as error:


            print(

                "DETECTION ERROR:",

                str(error)

            )


            time.sleep(1)


# ==========================================
# START DETECTION THREAD
# ==========================================

@app.on_event("startup")
def start_detection():

    print(
        "Starting detection thread..."
    )


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


        # Send a waiting frame instead of
        # leaving browser permanently loading

        if frame is None:


            blank = cv2.imread("")


            if blank is None:

                import numpy as np


                blank = np.zeros(

                    (
                        480,
                        854,
                        3
                    ),

                    dtype=np.uint8

                )


                cv2.putText(

                    blank,

                    "LOADING AI VIDEO...",

                    (
                        230,
                        240
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    1,

                    (
                        255,
                        255,
                        255
                    ),

                    2

                )


            frame = blank


        success, buffer = cv2.imencode(

            ".jpg",

            frame,

            [

                cv2.IMWRITE_JPEG_QUALITY,

                80

            ]

        )


        if success:


            frame_bytes = buffer.tobytes()


            yield (

                b"--frame\r\n"

                b"Content-Type: image/jpeg\r\n\r\n"

                +

                frame_bytes

                +

                b"\r\n"

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

        "multipart/x-mixed-replace; "

        "boundary=frame"

    )


# ==========================================
# STATUS
# ==========================================

@app.get("/status")
def status():

    with frame_lock:

        intruders = current_intruders

    files = []

    if os.path.exists(ALERTS_DIR):

        for file in os.listdir(ALERTS_DIR):

            if file.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                files.append(file)

    return {
        "status": "active",
        "current_intruders": int(intruders),
        "total_alerts": len(files)
    }