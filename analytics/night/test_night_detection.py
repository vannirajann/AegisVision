import sys
import os
import cv2

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analytics.video.video_source import VideoSource
from analytics.night.night_detector import NightDetector


print("Starting AegisVision night detection test...")

video_source = VideoSource(0)
night_detector = NightDetector()

video_source.open()

print("Camera opened successfully")
print("Live low-light detection started")
print("Press Q to quit")

try:

    while True:

        ret, frame = video_source.read()

        if not ret or frame is None:
            print("Could not read camera frame")
            break

        # Detect light condition
        result = night_detector.detect(frame)

        brightness = result["brightness"]
        status = result["status"]

        # Display brightness
        cv2.putText(
            frame,
            f"Brightness: {brightness}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # Display light status
        cv2.putText(
            frame,
            status,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "AegisVision - Night Detection",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    video_source.release()
    cv2.destroyAllWindows()

    print("Night detection test stopped")