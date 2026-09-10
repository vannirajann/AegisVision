from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import cv2
import os
import time

from config import ALERTS_DIR, STATIC_DIR
from multi_camera import multi_camera_manager


# ============================================================
# AEGISVISION FASTAPI SERVER
# ============================================================

app = FastAPI(
    title="AegisVision Detection Module",
    version="2.1"
)


# ============================================================
# STATIC FILES
# ============================================================

os.makedirs(STATIC_DIR, exist_ok=True)

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    index_path = os.path.join(
        STATIC_DIR,
        "index.html"
    )

    if os.path.exists(index_path):
        return FileResponse(index_path)

    return {
        "system": "AegisVision",
        "status": "running",
        "message": "Four-camera detection server is active"
    }


# ============================================================
# START DETECTION
# ============================================================

@app.on_event("startup")
def start_detection():

    print()
    print("?? Starting AegisVision Four-Camera Detection...")
    print()

    multi_camera_manager.start()


# ============================================================
# STOP DETECTION
# ============================================================

@app.on_event("shutdown")
def stop_detection():

    print()
    print("?? Stopping AegisVision...")

    multi_camera_manager.stop()

    print("? AegisVision stopped.")


# ============================================================
# BUILD CAMERA DATA
# ============================================================

def build_camera_data():

    cameras = {}

    for camera_id, camera in (
        multi_camera_manager.cameras.items()
    ):

        cameras[str(camera_id)] = {

            "camera_id": int(camera_id),

            "camera_name":
                camera.name,

            "frame_id":
                camera.frame_id,

            "running":
                camera.running,

            "detections":
                camera.current_detections,

            "current_objects":
                camera.current_objects,

            "current_persons":
                camera.current_persons,

            "current_vehicles":
                camera.current_vehicles,

            "current_dogs":
                camera.current_dogs,

            "total_objects":
                camera.total_objects,

            "total_persons":
                camera.total_persons,

            "total_vehicles":
                camera.total_vehicles,

            "total_dogs":
                camera.total_dogs,

            "total_intruders":
                camera.total_intruders
        }

    return cameras


# ============================================================
# GET ALL DETECTIONS
# ============================================================

@app.get("/detections")
def get_detections():

    cameras = build_camera_data()

    all_detections = []

    for camera_id, data in cameras.items():

        for detection in data["detections"]:

            detection_copy = detection.copy()

            detection_copy["camera_id"] = int(camera_id)

            detection_copy["camera_name"] = (
                data["camera_name"]
            )

            all_detections.append(
                detection_copy
            )

    frame_ids = [
        data["frame_id"]
        for data in cameras.values()
    ]

    return {

        "frame_id":
            max(frame_ids, default=0),

        "detections":
            all_detections,

        "cameras":
            cameras
    }


# ============================================================
# CAMERA STATUS
# ============================================================

@app.get("/status")
def get_status():

    cameras = build_camera_data()

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
            len(multi_camera_manager.cameras),

        "features": [

            "person_detection",

            "vehicle_detection",

            "dog_detection",

            "object_tracking",

            "individual_camera_fences",

            "virtual_fence",

            "intrusion_detection",

            "camera_specific_alerts",

            "evidence_capture",

            "multi_camera_monitoring",

            "live_video_streaming"

        ],

        "cameras":
            cameras
    }


# ============================================================
# GET ALERTS
# ============================================================

@app.get("/alerts")
def get_alerts():

    alert_directory = os.path.join(
        ALERTS_DIR,
        "multi_camera"
    )

    alerts = []

    if os.path.exists(alert_directory):

        for camera_folder in os.listdir(
            alert_directory
        ):

            camera_path = os.path.join(
                alert_directory,
                camera_folder
            )

            if not os.path.isdir(camera_path):
                continue

            for filename in os.listdir(
                camera_path
            ):

                if filename.lower().endswith(
                    (
                        ".jpg",
                        ".jpeg",
                        ".png"
                    )
                ):

                    alerts.append({
                        "camera":
                            camera_folder,

                        "filename":
                            filename,

                        "image_url":
                            f"/alert-images/{camera_folder}/{filename}"
                    })

    alerts.sort(
        key=lambda x: x["filename"],
        reverse=True
    )

    return {

        "total_alerts":
            len(alerts),

        "alerts":
            alerts
    }


# ============================================================
# SHOW ALERT IMAGE
# ============================================================

@app.get(
    "/alert-images/{camera_folder}/{filename}"
)
def get_alert_image(
    camera_folder: str,
    filename: str
):

    image_path = os.path.join(
        ALERTS_DIR,
        "multi_camera",
        camera_folder,
        filename
    )

    if os.path.exists(image_path):

        return FileResponse(
            image_path,
            media_type="image/jpeg"
        )

    return {
        "error": "Image not found"
    }


# ============================================================
# GENERATE CONTINUOUS MJPEG STREAM
# ============================================================

def generate_frames():

    print("?? Video stream client connected.")

    try:

        while True:

            frame = (
                multi_camera_manager
                .get_grid()
            )

            if frame is None:

                time.sleep(0.05)

                continue

            success, buffer = cv2.imencode(
                ".jpg",
                frame,
                [
                    int(cv2.IMWRITE_JPEG_QUALITY),
                    85
                ]
            )

            if not success:

                time.sleep(0.01)

                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: "
                + str(len(frame_bytes)).encode()
                + b"\r\n"
                b"Cache-Control: no-cache\r\n"
                b"\r\n"
                + frame_bytes
                + b"\r\n"
            )

            time.sleep(0.03)

    except GeneratorExit:

        print("?? Video stream client disconnected.")

    except Exception as error:

        print(
            f"? Video streaming error: {error}"
        )


# ============================================================
# FOUR-CAMERA LIVE VIDEO FEED
# ============================================================

@app.get("/video-feed")
def video_feed():

    return StreamingResponse(

        generate_frames(),

        media_type=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        ),

        headers={

            "Cache-Control":
                "no-cache, no-store, must-revalidate",

            "Pragma":
                "no-cache",

            "Expires":
                "0",

            "Connection":
                "keep-alive",

            "X-Accel-Buffering":
                "no"
        }
    )


# ============================================================
# STREAM ALIAS
# ============================================================

@app.get("/stream")
def stream():

    return StreamingResponse(

        generate_frames(),

        media_type=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        ),

        headers={

            "Cache-Control":
                "no-cache, no-store, must-revalidate",

            "Pragma":
                "no-cache",

            "Expires":
                "0",

            "Connection":
                "keep-alive",

            "X-Accel-Buffering":
                "no"
        }
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {

        "status": "ok",

        "service":
            "AegisVision Detection",

        "cameras":
            len(multi_camera_manager.cameras)
    }
