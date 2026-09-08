import cv2
import os
import sys

# Add analytics folder to Python path
ANALYTICS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if ANALYTICS_DIR not in sys.path:
    sys.path.insert(0, ANALYTICS_DIR)

from face.face_detector import FaceDetector


VIDEO_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "detection",
        "videos",
        "test.mp4"
    )
)


print("Starting Face Detection...")
print("Video:", VIDEO_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Cannot open video")
    sys.exit()

detector = FaceDetector()

print("Face detector loaded successfully")
print("Processing video...")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video finished.")
        break

    faces = detector.detect_faces(frame)

    print("Faces detected:", len(faces))

    for face in faces:

        x = face["x"]
        y = face["y"]
        width = face["width"]
        height = face["height"]

        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "FACE",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "AegisVision - Face Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

print("Face detection test completed.")