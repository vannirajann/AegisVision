import cv2
import sys
import os
from ultralytics import YOLO

from loitering_detector import LoiteringDetector

VIDEO_PATH = r"D:\AegisVision\detection\videos\test.mp4"
MODEL_PATH = r"D:\AegisVision\detection\yolo11n.pt"

print("Starting Loitering Detection...")
print("Video:", VIDEO_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Cannot open video")
    sys.exit()

print("Loading YOLO person tracker...")

model = YOLO(MODEL_PATH)

loitering_detector = LoiteringDetector(threshold_seconds=10)

print("YOLO tracker loaded")
print("Loitering detector loaded")
print("Processing video...")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video finished.")
        break

    results = model.track(
        frame,
        persist=True,
        classes=[0],
        conf=0.30,
        tracker="bytetrack.yaml",
        verbose=False
    )

    current_ids = set()

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, track_ids):

            x1, y1, x2, y2 = map(int, box)

            current_ids.add(track_id)

            loitering_detector.person_entered(track_id)

            status = loitering_detector.get_status(track_id)

            label = f"ID:{track_id} {status['duration_seconds']}s"

            if status["loitering"]:
                label += " LOITERING"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    cv2.imshow(
        "AegisVision - Loitering Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Loitering video test completed.")