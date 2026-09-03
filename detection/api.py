from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import cv2
import os
from datetime import datetime
import threading
import time

from config import (
    ALERTS_DIR,
    STATIC_DIR,
    VIDEO_PATH
)

from detector import (
    detect_objects,
    create_detection_output
)


# ==========================================
# FASTAPI APP
# ==========================================

app = FastAPI(
    title="AegisVision Detection Module",
    version="1.0"
)


# ==========================================
# GLOBAL VARIABLES
# ==========================================

latest_frame = None

frame_lock = threading.Lock()


# Previous position of tracked objects

previous_positions = {}


# Objects already counted for alerts

counted_objects = set()


# Latest structured detection output

latest_detection_output = {
    "frame_id": 0,
    "detections": []
}

detection_lock = threading.Lock()


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
# HOME DASHBOARD
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
# GET ALERTS
# ==========================================

@app.get("/alerts")
def get_alerts():

    files = []

    if os.path.exists(ALERTS_DIR):

        for file in os.listdir(ALERTS_DIR):

            if file.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png"
                )
            ):

                files.append(file)

    return {

        "total_alerts": len(files),

        "alerts": sorted(
            files,
            reverse=True
        )

    }


# ==========================================
# SHOW ALERT IMAGE
# ==========================================

@app.get(
    "/alert-images/{filename}"
)
def get_alert_image(
    filename: str
):

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
# GET STRUCTURED DETECTION OUTPUT
# ==========================================

@app.get("/detections")
def get_detections():

    with detection_lock:

        return latest_detection_output


# ==========================================
# SYSTEM STATUS
# ==========================================

@app.get("/status")
def get_status():

    return {

        "module":
        "AegisVision Detection",

        "status":
        "active",

        "model":
        "YOLO11n",

        "source":
        "video",

        "features": [

            "person_detection",

            "vehicle_detection",

            "object_tracking",

            "restricted_zone",

            "intrusion_detection",

            "evidence_capture"

        ]

    }


# ==========================================
# DETECTION LOOP
# ==========================================

def detection_loop():

    global latest_frame
    global latest_detection_output


    # ==========================================
    # OPEN VIDEO
    # ==========================================

    video = cv2.VideoCapture(
        VIDEO_PATH
    )


    if not video.isOpened():

        print(
            "❌ ERROR: Cannot open video:"
        )

        print(
            VIDEO_PATH
        )

        return


    print(
        "✅ Video source connected:"
    )

    print(
        VIDEO_PATH
    )


    frame_id = 0


    # ==========================================
    # MAIN LOOP
    # ==========================================

    while True:


        # ==========================================
        # READ VIDEO FRAME
        # ==========================================

        success, frame = video.read()


        # ==========================================
        # RESTART VIDEO
        # ==========================================

        if not success:

            print(
                "🔄 Video finished. Restarting..."
            )

            video.set(
                cv2.CAP_PROP_POS_FRAMES,
                0
            )

            previous_positions.clear()

            # Keep counted objects
            # to prevent duplicate alerts

            continue


        # ==========================================
        # FRAME ID
        # ==========================================

        frame_id += 1


        # ==========================================
        # FLIP VIDEO
        # ==========================================

        frame = cv2.flip(
            frame,
            1
        )


        # ==========================================
        # FRAME DIMENSIONS
        # ==========================================

        height, width = frame.shape[:2]


        # ==========================================
        # RESTRICTED ZONE
        # ==========================================

        ZONE_Y = height // 2


        # ==========================================
        # YOLO DETECTION + TRACKING
        # ==========================================

        results = detect_objects(
            frame
        )


        # ==========================================
        # CREATE OUTPUT FRAME
        # ==========================================

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


        # ==========================================
        # DRAW INTRUSION LINE
        # ==========================================

        cv2.line(

            output_frame,

            (0, ZONE_Y),

            (width, ZONE_Y),

            (0, 0, 255),

            5

        )


        # ==========================================
        # RESTRICTED ZONE TEXT
        # ==========================================

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
        # CREATE STRUCTURED OUTPUT
        # ==========================================

        structured_output = (
            create_detection_output(

                results,

                frame_id,

                ZONE_Y

            )
        )


        # ==========================================
        # CHECK EVERY DETECTION
        # ==========================================

        for detection in structured_output[
            "detections"
        ]:


            class_name = detection[
                "class"
            ]


            confidence = detection[
                "confidence"
            ]


            track_id = detection[
                "track_id"
            ]


            bbox = detection[
                "bbox"
            ]


            x1 = bbox["x1"]
            y1 = bbox["y1"]
            x2 = bbox["x2"]
            y2 = bbox["y2"]


            # Object center

            center_x = (
                x1 + x2
            ) // 2


            center_y = (
                y1 + y2
            ) // 2


            # ======================================
            # DEFAULT SAFE
            # ======================================

            color = (
                0,
                255,
                0
            )


            label = (

                f"{class_name.upper()} "

                f"{confidence:.2f}"

            )


            if track_id is not None:

                label += (
                    f" ID:{track_id}"
                )


            # ======================================
            # OBJECT KEY
            # ======================================

            if track_id is not None:

                object_key = (
                    f"{class_name}_"
                    f"{track_id}"
                )

            else:

                object_key = (
                    f"{class_name}_"
                    f"{x1}_"
                    f"{y1}"
                )


            # ======================================
            # CHECK LINE CROSSING
            # ======================================

            crossed_line = False


            previous_y = previous_positions.get(
                object_key
            )


            if previous_y is not None:

                if (

                    previous_y > ZONE_Y

                    and

                    center_y <= ZONE_Y

                ):

                    crossed_line = True


            # Save current position

            previous_positions[
                object_key
            ] = center_y


            # ======================================
            # CHECK INTRUDER
            # ======================================

            if detection["intruder"]:


                color = (
                    0,
                    0,
                    255
                )


                label = (

                    f"INTRUDER: "

                    f"{class_name.upper()} "

                    f"{confidence:.2f}"

                )


                if track_id is not None:

                    label += (
                        f" ID:{track_id}"
                    )


            # ======================================
            # CREATE ALERT
            # ======================================

            if crossed_line:


                position_key = (

                    f"{class_name}_"

                    f"{center_x // 150}"

                )


                if position_key not in counted_objects:


                    counted_objects.add(
                        position_key
                    )


                    timestamp = (
                        datetime.now().strftime(
                            "%Y%m%d_%H%M%S_%f"
                        )
                    )


                    filename = (

                        f"intruder_"

                        f"{class_name}_"

                        f"ID_{track_id}_"

                        f"{timestamp}.jpg"

                    )


                    file_path = os.path.join(

                        ALERTS_DIR,

                        filename

                    )


                    # Save evidence

                    cv2.imwrite(

                        file_path,

                        output_frame

                    )


                    print(
                        "\n🚨 INTRUSION DETECTED!"
                    )


                    print(
                        f"Object: "
                        f"{class_name.upper()}"
                    )


                    print(
                        f"Confidence: "
                        f"{confidence:.2f}"
                    )


                    print(
                        f"ID: "
                        f"{track_id}"
                    )


                    print(
                        f"Evidence: "
                        f"{file_path}\n"
                    )


            # ======================================
            # DRAW OBJECT BOX
            # ======================================

            cv2.rectangle(

                output_frame,

                (x1, y1),

                (x2, y2),

                color,

                3

            )


            # ======================================
            # DRAW LABEL
            # ======================================

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


        # ==========================================
        # UPDATE DETECTION OUTPUT
        # ==========================================

        with detection_lock:

            latest_detection_output = (
                structured_output
            )


        # ==========================================
        # STORE LATEST FRAME
        # ==========================================

        with frame_lock:

            latest_frame = (
                output_frame.copy()
            )


# ==========================================
# START DETECTION
# ==========================================

@app.on_event("startup")
def start_detection():

    print(
        "🚀 Starting AegisVision..."
    )


    thread = threading.Thread(

        target=detection_loop,

        daemon=True

    )


    thread.start()


# ==========================================
# VIDEO STREAM GENERATOR
# ==========================================

def generate_frames():

    while True:


        with frame_lock:


            if latest_frame is None:

                frame = None


            else:

                frame = (
                    latest_frame.copy()
                )


        if frame is None:

            time.sleep(
                0.05
            )

            continue


        success, buffer = cv2.imencode(

            ".jpg",

            frame

        )


        if not success:

            continue


        frame_bytes = (
            buffer.tobytes()
        )


        yield (

            b"--frame\r\n"

            b"Content-Type: image/jpeg\r\n\r\n"

            + frame_bytes

            + b"\r\n"

        )


        time.sleep(
            0.03
        )


# ==========================================
# LIVE VIDEO FEED
# ==========================================

@app.get("/video-feed")
def video_feed():

    return StreamingResponse(

        generate_frames(),

        media_type=(

            "multipart/x-mixed-replace; "

            "boundary=frame"

        )

    )