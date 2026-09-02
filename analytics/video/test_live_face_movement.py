import cv2
import sys
import os
import time

# Project root
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

sys.path.insert(0, project_root)

from analytics.video.video_source import VideoSource
from analytics.face.face_detector import FaceDetector
from analytics.movement.movement_detector import MovementDetector


# Create components
source = VideoSource(0)
face_detector = FaceDetector()
movement_detector = MovementDetector(threshold=3.0)


try:
    source.open()

    print("Camera opened successfully")
    print("AegisVision live detection started")
    print("Move your hand or body")
    print("Press Q to quit")

    last_face_detection = 0
    faces = []

    while True:

        # Read camera frame
        ret, frame = source.read()

        if not ret or frame is None:
            print("ERROR: Camera frame not received")
            break

        # -------------------------------------------------
        # MOVEMENT DETECTION
        # Runs on every frame - very fast
        # -------------------------------------------------

        movement_detected = movement_detector.detect_movement(frame)

        # -------------------------------------------------
        # FACE DETECTION
        # Run only every 5 frames to reduce lag
        # -------------------------------------------------

        current_time = time.time()

        if current_time - last_face_detection > 0.15:

            faces = face_detector.detect_faces(frame)

            last_face_detection = current_time

        # -------------------------------------------------
        # DRAW FACE BOXES
        # -------------------------------------------------

        for face in faces:

            x = face["x"]
            y = face["y"]
            w = face["width"]
            h = face["height"]

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        # -------------------------------------------------
        # DISPLAY FACE COUNT
        # -------------------------------------------------

        cv2.putText(
            frame,
            f"Faces: {len(faces)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # -------------------------------------------------
        # DISPLAY MOVEMENT
        # -------------------------------------------------

        if movement_detected:
            movement_text = "MOVEMENT DETECTED"
        else:
            movement_text = "NO MOVEMENT"

        cv2.putText(
            frame,
            movement_text,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # -------------------------------------------------
        # DISPLAY VIDEO
        # -------------------------------------------------

        cv2.imshow(
            "AegisVision - Live Detection",
            frame
        )

        # Q = quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


finally:

    source.release()
    cv2.destroyAllWindows()

    print("Camera released")