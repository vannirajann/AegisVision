import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analytics.night.night_detector import NightDetector

print("Starting AegisVision night detection test...")

night_detector = NightDetector()
height, width = 240, 320

dark_frame = np.zeros((height, width, 3), dtype=np.uint8)
light_frame = np.full((height, width, 3), 220, dtype=np.uint8)

for label, frame in [("dark", dark_frame), ("light", light_frame)]:
    result = night_detector.detect(frame)
    print(
        f"{label.upper()} FRAME -> brightness={result['brightness']}, "
        f"low_light={result['low_light']}, status={result['status']}"
    )

print("Night detection test completed")