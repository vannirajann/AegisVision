import cv2
import sys
import os

# Add project root to Python path
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, project_root)

from analytics.video.video_source import VideoSource
from analytics.face.face_detector import FaceDetector


source = VideoSource(0)
detector = FaceDetector()

try:
    source.open()

    print("Camera opened successfully")
    print("Live face detection started")
    print("Press Q to quit")

    while True:
        ret, frame = source.read()

        if not ret or frame is None:
            print("ERROR: Could not read frame")
            break

        # Detect faces
        faces = detector.detect_faces(frame)

        # Draw detected faces
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

        # Display number of faces
        cv2.putText(
            frame,
            f"Faces: {len(faces)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "AegisVision - Live Face Detection",
            frame
        )

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    source.release()
    cv2.destroyAllWindows()
    print("Camera released")