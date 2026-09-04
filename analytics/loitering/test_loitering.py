import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analytics.loitering.loitering_detector import LoiteringDetector

print("Starting loitering detection test...")

detector = LoiteringDetector(threshold_seconds=10)
track_id = 1

print("Person entered restricted zone")
detector.person_entered(track_id)

for i in range(15):
    status = detector.get_status(track_id)
    print(
        f"Track ID: {status['track_id']} | "
        f"Duration: {status['duration_seconds']} sec | "
        f"Loitering: {status['loitering']}"
    )
    time.sleep(1)

print("Loitering test completed")