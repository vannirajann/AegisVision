import cv2
import os
import sys

from movement_detector import MovementDetector

VIDEO_PATH = r"D:\AegisVision\detection\videos\test.mp4"

print("Starting Movement Detection...")
print("Video:", VIDEO_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Cannot open video")
    sys.exit()

detector = MovementDetector(threshold=3.0)

print("Movement detector loaded successfully")
print("Processing video...")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video finished.")
        break

    change = detector.get_change_amount(frame)
    movement = change > 3.0

    if movement:
        status = "MOVEMENT DETECTED"
    else:
        status = "NO MOVEMENT"

    cv2.putText(
        frame,
        status,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Change: {change:.2f}",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("AegisVision - Movement Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Movement detection test completed.")