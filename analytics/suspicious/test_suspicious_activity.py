import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analytics.suspicious.suspicious_activity_detector import SuspiciousActivityDetector

print("Starting suspicious activity detection test...")

detector = SuspiciousActivityDetector()

print("\nTEST 1: Loitering")
result = detector.analyze(track_id=1, loitering=True)
print(result)

print("\nTEST 2: Repeated Entry")
result = detector.analyze(track_id=2, repeated_entry=True)
print(result)

print("\nTEST 3: Night Movement")
result = detector.analyze(track_id=3, night_movement=True)
print(result)

print("\nTEST 4: Multiple Suspicious Conditions")
result = detector.analyze(track_id=4, loitering=True, repeated_entry=True, night_movement=True)
print(result)

print("\nTEST 5: Normal Activity")
result = detector.analyze(track_id=5)
print(result)

print("\nSuspicious activity test completed")