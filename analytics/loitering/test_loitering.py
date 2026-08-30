import time

from loitering_detector import LoiteringDetector


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