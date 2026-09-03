from ultralytics import YOLO

from config import (
    MODEL_PATH,
    ALLOWED_CLASSES
)


# ==========================================
# LOAD YOLO MODEL
# ==========================================

model = YOLO(MODEL_PATH)


# ==========================================
# DETECT AND TRACK OBJECTS
# ==========================================

def detect_objects(frame):

    results = model.track(
        frame,
        persist=True,
        classes=ALLOWED_CLASSES,
        tracker="bytetrack.yaml",
        verbose=False
    )

    return results


# ==========================================
# CREATE STRUCTURED DETECTION OUTPUT
# ==========================================

def create_detection_output(
    results,
    frame_id,
    zone_y
):

    detections = []

    for box in results[0].boxes:

        class_id = int(box.cls[0])

        class_name = model.names[class_id]

        confidence = round(
            float(box.conf[0]),
            2
        )

        x1, y1, x2, y2 = box.xyxy[0]

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        # Get tracking ID
        if box.id is not None:

            track_id = int(box.id[0])

        else:

            track_id = None


        # Object center
        center_y = (y1 + y2) // 2


        # Check restricted zone
        intruder = center_y < zone_y


        detection = {

            "class": class_name,

            "confidence": confidence,

            "track_id": track_id,

            "bbox": {

                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2

            },

            "intruder": intruder

        }


        detections.append(
            detection
        )


    return {

        "frame_id": frame_id,

        "detections": detections

    }