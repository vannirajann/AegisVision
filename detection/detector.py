from ultralytics import YOLO

from config import (
    MODEL_PATH,
    ALLOWED_CLASSES
)


# ==========================================
# LOAD YOLO MODEL
# ==========================================

model = YOLO(
    MODEL_PATH
)


# ==========================================
# DETECT + TRACK OBJECTS
# ==========================================

def detect_objects(
    frame
):

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
    fence=None,
    restricted_side=1
):

    detections = []


    for box in results[0].boxes:

        # ==================================
        # CLASS
        # ==================================

        class_id = int(
            box.cls[0]
        )

        class_name = model.names[
            class_id
        ]


        # ==================================
        # CONFIDENCE
        # ==================================

        confidence = round(

            float(
                box.conf[0]
            ),

            2

        )


        # ==================================
        # BOUNDING BOX
        # ==================================

        x1, y1, x2, y2 = box.xyxy[0]

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)


        # ==================================
        # TRACK ID
        # ==================================

        if box.id is not None:

            track_id = int(
                box.id[0]
            )

        else:

            track_id = None


        # ==================================
        # CENTER
        # ==================================

        center_x = (
            x1 + x2
        ) // 2

        center_y = (
            y1 + y2
        ) // 2


        # ==================================
        # DEFAULT
        # ==================================

        intruder = False


        # ==================================
        # FENCE CHECK
        # ==================================

        if fence is not None:

            height, width = (
                results[0].orig_shape
            )

            p1 = (
                int(fence[0][0] * width),
                int(fence[0][1] * height)
            )

            p2 = (
                int(fence[1][0] * width),
                int(fence[1][1] * height)
            )


            # Bottom-center of object
            # is better for ground crossing

            point_x = center_x
            point_y = y2


            # Cross product
            side = (

                (p2[0] - p1[0])
                *
                (point_y - p1[1])

                -

                (p2[1] - p1[1])
                *
                (point_x - p1[0])

            )


            if side > 0:

                current_side = 1

            elif side < 0:

                current_side = -1

            else:

                current_side = 0


            if (
                current_side
                == restricted_side
            ):

                intruder = True


        # ==================================
        # STRUCTURED DETECTION
        # ==================================

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

            "center": {

                "x": center_x,
                "y": center_y

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