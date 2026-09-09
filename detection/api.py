from fastapi import FastAPI
from fastapi.responses import (
    FileResponse,
    StreamingResponse
)
from fastapi.staticfiles import StaticFiles

import cv2
import os
import time

from config import (
    ALERTS_DIR,
    STATIC_DIR
)

from multi_camera import (
    multi_camera_manager
)


# ==========================================
# FASTAPI APP
# ==========================================

app = FastAPI(

    title="AegisVision Detection Module",

    version="2.0"

)


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
# START DETECTION
# ==========================================

@app.on_event("startup")
def start_detection():

    print(
        "🚀 Starting AegisVision "
        "Four-Camera Detection..."
    )

    multi_camera_manager.start()


# ==========================================
# STOP DETECTION
# ==========================================

@app.on_event("shutdown")
def stop_detection():

    multi_camera_manager.stop()


# ==========================================
# GET ALL DETECTIONS
# ==========================================

@app.get("/detections")
def get_detections():

    cameras = (
        multi_camera_manager
        .get_detections()
    )


    # Combine detections so the
    # existing API structure remains useful

    all_detections = []


    for camera_id, data in (
        cameras.items()
    ):

        for detection in (
            data["detections"]
        ):

            detection_copy = (
                detection.copy()
            )

            detection_copy[
                "camera_id"
            ] = int(camera_id)

            detection_copy[
                "camera_name"
            ] = data[
                "camera_name"
            ]

            all_detections.append(
                detection_copy
            )


    return {

        "frame_id": max(

            [
                data["frame_id"]
                for data in cameras.values()
            ],

            default=0

        ),

        "detections":
        all_detections,

        "cameras":
        cameras

    }


# ==========================================
# CAMERA STATUS
# ==========================================

@app.get("/status")
def get_status():

    cameras = (
        multi_camera_manager
        .get_detections()
    )


    return {

        "module":
        "AegisVision Detection",

        "status":
        "active",

        "model":
        "YOLO11n",

        "source":
        "4 video cameras",

        "camera_count":
        4,

        "features": [

            "person_detection",

            "vehicle_detection",

            "object_tracking",

            "individual_camera_fences",

            "intrusion_detection",

            "camera_specific_alerts",

            "evidence_capture",

            "multi_camera_monitoring"

        ],

        "cameras":
        cameras

    }


# ==========================================
# GET ALERTS
# ==========================================

@app.get("/alerts")
def get_alerts():

    alert_directory = os.path.join(

        ALERTS_DIR,

        "multi_camera"

    )


    files = []


    if os.path.exists(
        alert_directory
    ):

        for file in os.listdir(
            alert_directory
        ):

            if file.lower().endswith(

                (
                    ".jpg",
                    ".jpeg",
                    ".png"
                )

            ):

                files.append(file)


    return {

        "total_alerts":
        len(files),

        "alerts":
        sorted(
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

        "multi_camera",

        filename

    )


    if os.path.exists(
        image_path
    ):

        return FileResponse(
            image_path
        )


    return {

        "error":
        "Image not found"

    }


# ==========================================
# GENERATE 4-CAMERA STREAM
# ==========================================

def generate_frames():

    while True:

        frame = (
            multi_camera_manager
            .get_grid()
        )


        if frame is None:

            time.sleep(
                0.05
            )

            continue


        success, buffer = (
            cv2.imencode(
                ".jpg",
                frame
            )
        )


        if not success:

            continue


        frame_bytes = (
            buffer.tobytes()
        )


        yield (

            b"--frame\r\n"

            b"Content-Type: "
            b"image/jpeg\r\n\r\n"

            + frame_bytes

            + b"\r\n"

        )


        time.sleep(
            0.03
        )


# ==========================================
# FOUR-CAMERA VIDEO FEED
# ==========================================

@app.get("/video-feed")
def video_feed():

    return StreamingResponse(

        generate_frames(),

        media_type=(

            "multipart/"
            "x-mixed-replace;"
            " boundary=frame"

        )

    )