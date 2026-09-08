import cv2
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "movement"))

from night_detector import NightDetector
from movement_detector import MovementDetector


VIDEO_PATH = r"D:\AegisVision\detection\videos\test.mp4"

print("Starting Night + Movement Detection...")
print("Video:", VIDEO_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Cannot open video")
    sys.exit()

night_detector = NightDetector(brightness_threshold=80)
movement_detector = MovementDetector(threshold=3.0)

print("Night detector loaded")
print("Movement detector loaded")
print("Processing video...")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video finished.")
        break

    night_result = night_detector.detect(frame)
    movement = movement_detector.detect_movement(frame)

    if night_result["low_light"] and movement:
        status = "NIGHT MOVEMENT DETECTED"
    elif night_result["low_light"]:
        status = "NIGHT / LOW LIGHT"
    elif movement:
        status = "MOVEMENT DETECTED"
    else:
        status = "NORMAL"

    cv2.putText(
        frame,
        status,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Brightness: {night_result['brightness']:.2f}",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("AegisVision - Night Movement", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Night + Movement test completed.")