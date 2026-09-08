import cv2
import sys

from night_detector import NightDetector

VIDEO_PATH = r"D:\AegisVision\detection\videos\test.mp4"

print("Starting Night Detection...")
print("Video:", VIDEO_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Cannot open video")
    sys.exit()

detector = NightDetector(brightness_threshold=80)

print("Night detector loaded successfully")
print("Processing video...")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video finished.")
        break

    result = detector.detect(frame)

    status = result["status"]
    brightness = result["brightness"]

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
        f"Brightness: {brightness}",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("AegisVision - Night Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Night detection test completed.")