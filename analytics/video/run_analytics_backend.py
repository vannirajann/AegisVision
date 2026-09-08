import cv2
import os
import sys
import time

# =========================================================
# PATH SETUP
# =========================================================

ANALYTICS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

PROJECT_DIR = os.path.abspath(
    os.path.join(ANALYTICS_DIR, "..")
)

sys.path.insert(0, ANALYTICS_DIR)
sys.path.insert(0, PROJECT_DIR)

# =========================================================
# IMPORTS
# =========================================================

from face.face_detector import FaceDetector
from movement.movement_detector import MovementDetector
from night.night_detector import NightDetector
from loitering.loitering_detector import LoiteringDetector
from suspicious.suspicious_activity_detector import SuspiciousActivityDetector
from events.analytics_interface import AnalyticsInterface

from ultralytics import YOLO

# =========================================================
# FILE PATHS
# =========================================================

VIDEO_PATH = os.path.join(
    PROJECT_DIR,
    "detection",
    "videos",
    "test.mp4"
)

YOLO_MODEL = os.path.join(
    PROJECT_DIR,
    "detection",
    "yolo11n.pt"
)

# =========================================================
# START
# =========================================================

print("===================================")
print("AegisVision Analytics Integration")
print("===================================")
print("Video:", VIDEO_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Cannot open video")
    sys.exit()

print("Video opened successfully")

# =========================================================
# ANALYTICS MODULES
# =========================================================

face_detector = FaceDetector()
movement_detector = MovementDetector()
night_detector = NightDetector()

loitering_detector = LoiteringDetector(
    threshold_seconds=10
)

suspicious_detector = SuspiciousActivityDetector()

analytics = AnalyticsInterface()

# =========================================================
# YOLO TRACKING
# =========================================================

print("Loading YOLO tracking model...")

model = YOLO(YOLO_MODEL)

print("YOLO tracking model loaded")

# =========================================================
# EVENT CONTROL
# =========================================================

last_face_event = 0
last_night_event = 0

loitering_sent = set()

# =========================================================
# VIDEO PROCESSING
# =========================================================

print("Processing video...")
print("Press Q to stop")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video finished.")
        break

    current_time = time.time()

    # =====================================================
    # FACE DETECTION - YuNet
    # =====================================================

    faces = face_detector.detect_faces(frame)

    # -----------------------------------------------------
    # DRAW GREEN FACE BOXES
    # -----------------------------------------------------

    for face in faces:

        x = int(face["x"])
        y = int(face["y"])
        width = int(face["width"])
        height = int(face["height"])

        # Green rectangle around detected face
        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2
        )

        # FACE label
        cv2.putText(
            frame,
            "FACE",
            (x, max(y - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # -----------------------------------------------------
    # SEND FACE EVENT
    # -----------------------------------------------------

    if (
        len(faces) > 0
        and current_time - last_face_event > 5
    ):

        analytics.create_face_event(
            track_id=None,
            face_count=len(faces)
        )

        print(
            "FACE EVENT:",
            len(faces),
            "face(s)"
        )

        last_face_event = current_time

    # =====================================================
    # MOVEMENT DETECTION
    # =====================================================

    movement = movement_detector.detect_movement(frame)

    # =====================================================
    # NIGHT DETECTION
    # =====================================================

    night_result = night_detector.detect(frame)

    # =====================================================
    # NIGHT MOVEMENT EVENT
    # =====================================================

    if (
        movement
        and night_result["low_light"]
        and current_time - last_night_event > 10
    ):

        analytics.create_night_movement_event(
            track_id=None,
            brightness=night_result["brightness"],
            light_status=night_result["status"]
        )

        print("NIGHT MOVEMENT EVENT")

        last_night_event = current_time

    # =====================================================
    # PERSON TRACKING
    # =====================================================

    results = model.track(
        frame,
        persist=True,
        classes=[0],
        conf=0.30,
        tracker="bytetrack.yaml",
        verbose=False
    )

    # =====================================================
    # DRAW PERSON TRACKING BOXES
    # =====================================================

    if results and results[0].boxes.id is not None:

        boxes = results[0].boxes

        track_ids = boxes.id.int().cpu().tolist()

        xyxy = boxes.xyxy.cpu().tolist()

        for box, track_id in zip(xyxy, track_ids):

            x1, y1, x2, y2 = map(int, box)

            # Blue person tracking box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            # Tracking ID
            cv2.putText(
                frame,
                f"ID: {track_id}",
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

            # ------------------------------------------------
            # LOITERING
            # ------------------------------------------------

            loitering_detector.person_entered(track_id)

            status = loitering_detector.get_status(track_id)

            if status["loitering"]:

                # Show loitering on screen
                cv2.putText(
                    frame,
                    "LOITERING",
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

                # Send only once per tracking ID
                if track_id not in loitering_sent:

                    analytics.create_loitering_event(
                        track_id,
                        status["duration_seconds"]
                    )

                    print(
                        "LOITERING EVENT:",
                        "ID",
                        track_id,
                        "Duration:",
                        status["duration_seconds"]
                    )

                    loitering_sent.add(track_id)

    # =====================================================
    # DISPLAY INFORMATION
    # =====================================================

    cv2.putText(
        frame,
        f"Faces: {len(faces)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Movement: {'YES' if movement else 'NO'}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        night_result["status"],
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    # =====================================================
    # SHOW VIDEO
    # =====================================================

    



# =========================================================
# CLEANUP
# =========================================================

cap.release()
try:
    cv2.destroyAllWindows()
except cv2.error:
    pass

print("===================================")
print("Analytics processing completed")
print("===================================")