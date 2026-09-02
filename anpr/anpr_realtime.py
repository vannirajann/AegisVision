"""
============================================================
AEGISVISION - ANPR
PHASE 6.2 - REAL-TIME NUMBER PLATE DETECTION
============================================================

Pipeline:
    Webcam
       ↓
    YOLO Plate Detector
       ↓
    Plate Bounding Box
       ↓
    Live Detection Display

Controls:
    Q / ESC -> Exit
    S       -> Save current frame
"""

import cv2
import os
import time
from datetime import datetime
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_SOURCE = 0

MODEL_PATH = os.path.join(
    "models",
    "plate_detector.pt"
)

WINDOW_NAME = "AegisVision - Real-Time ANPR"

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

CONFIDENCE_THRESHOLD = 0.25

OUTPUT_DIR = os.path.join(
    "output",
    "phase6_2"
)

CAPTURE_DIR = os.path.join(
    OUTPUT_DIR,
    "captured_frames"
)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CAPTURE_DIR, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("AEGISVISION - ANPR")
print("PHASE 6.2 - REAL-TIME NUMBER PLATE DETECTION")
print("=" * 60)


# ============================================================
# CHECK MODEL
# ============================================================

print("\nChecking YOLO model...")

if not os.path.exists(MODEL_PATH):
    print("\nERROR: YOLO model not found.")
    print(f"Expected path: {os.path.abspath(MODEL_PATH)}")
    raise SystemExit(1)

print(f"Model found: {os.path.abspath(MODEL_PATH)}")


# ============================================================
# LOAD YOLO MODEL
# ============================================================

print("\nLoading plate detection model...")

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print("\nERROR: Could not load YOLO model.")
    print(f"Details: {e}")
    raise SystemExit(1)

print("YOLO model loaded successfully.")


# ============================================================
# OPEN CAMERA
# ============================================================

print("\nOpening webcam...")

cap = cv2.VideoCapture(CAMERA_SOURCE)

if not cap.isOpened():
    print("\nERROR: Could not open webcam.")
    print("Try changing CAMERA_SOURCE from 0 to 1.")
    raise SystemExit(1)


# ============================================================
# CAMERA INFORMATION
# ============================================================

camera_width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

camera_height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

camera_fps = cap.get(
    cv2.CAP_PROP_FPS
)

if camera_fps <= 0:
    camera_fps = 30.0


print("\nWebcam opened successfully.")
print("-" * 60)
print(f"Resolution : {camera_width} x {camera_height}")
print(f"FPS        : {camera_fps:.2f}")
print("-" * 60)


# ============================================================
# PROCESSING VARIABLES
# ============================================================

frame_count = 0
total_detections = 0
saved_frames = 0

start_time = time.time()

display_fps = 0.0

fps_start_time = time.time()
fps_start_frame = 0


# ============================================================
# START REAL-TIME DETECTION
# ============================================================

print("\nREAL-TIME PLATE DETECTION STARTED")
print("-" * 60)
print("Point your camera toward a vehicle/number plate.")
print("Press Q or ESC to exit.")
print("Press S to save the current frame.")
print("-" * 60)


while True:

    # ========================================================
    # READ CAMERA FRAME
    # ========================================================

    ret, frame = cap.read()

    if not ret:
        print("\nERROR: Failed to read camera frame.")
        break

    frame_count += 1


    # ========================================================
    # YOLO PLATE DETECTION
    # ========================================================

    try:

        results = model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )

    except Exception as e:

        print(f"\nDetection error: {e}")
        continue


    # ========================================================
    # COPY FRAME FOR DISPLAY
    # ========================================================

    display_frame = frame.copy()


    # ========================================================
    # PROCESS DETECTIONS
    # ========================================================

    frame_detections = 0

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            confidence = float(
                box.conf[0]
            )

            # ------------------------------------------------
            # BOUNDING BOX
            # ------------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            # ------------------------------------------------
            # KEEP BOX INSIDE IMAGE
            # ------------------------------------------------

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(
                frame.shape[1],
                x2
            )

            y2 = min(
                frame.shape[0],
                y2
            )

            # ------------------------------------------------
            # DRAW PLATE BOX
            # ------------------------------------------------

            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            # ------------------------------------------------
            # LABEL
            # ------------------------------------------------

            label = (
                f"NUMBER PLATE "
                f"{confidence * 100:.1f}%"
            )

            label_y = max(
                30,
                y1 - 10
            )

            cv2.putText(
                display_frame,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            frame_detections += 1


    total_detections += frame_detections


    # ========================================================
    # CALCULATE DISPLAY FPS
    # ========================================================

    current_time = time.time()

    elapsed = (
        current_time -
        fps_start_time
    )

    if elapsed >= 1.0:

        display_fps = (
            frame_count -
            fps_start_frame
        ) / elapsed

        fps_start_frame = frame_count
        fps_start_time = current_time


    # ========================================================
    # RESIZE DISPLAY
    # ========================================================

    display_frame = cv2.resize(
        display_frame,
        (
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT
        )
    )


    # ========================================================
    # INFORMATION PANEL
    # ========================================================

    cv2.putText(
        display_frame,
        "AEGISVISION - REAL-TIME ANPR",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        display_frame,
        f"Frame: {frame_count}",
        (25, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        display_frame,
        f"FPS: {display_fps:.1f}",
        (25, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        display_frame,
        f"Plates detected: {frame_detections}",
        (25, 141),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        display_frame,
        "DETECTOR: ACTIVE",
        (25, 174),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # CONTROLS
    # ========================================================

    cv2.putText(
        display_frame,
        "Q / ESC = Exit    S = Save Frame",
        (
            25,
            DISPLAY_HEIGHT - 25
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # SHOW WINDOW
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        display_frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # ========================================================
    # SAVE FRAME
    # ========================================================

    if key == ord("s"):

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )[:-3]

        filename = (
            f"frame_{frame_count}_"
            f"{timestamp}.jpg"
        )

        filepath = os.path.join(
            CAPTURE_DIR,
            filename
        )

        cv2.imwrite(
            filepath,
            display_frame
        )

        saved_frames += 1

        print(
            f"Frame saved: {filepath}"
        )


    # ========================================================
    # EXIT
    # ========================================================

    if key == ord("q") or key == 27:
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()


# ============================================================
# FINAL STATISTICS
# ============================================================

end_time = time.time()

processing_time = (
    end_time -
    start_time
)

if processing_time > 0:

    average_fps = (
        frame_count /
        processing_time
    )

else:

    average_fps = 0.0


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("PHASE 6.2 SUMMARY")
print("=" * 60)

print(
    f"Frames processed : {frame_count}"
)

print(
    f"Processing time  : "
    f"{processing_time:.2f} seconds"
)

print(
    f"Average FPS      : "
    f"{average_fps:.2f}"
)

print(
    f"Total detections : "
    f"{total_detections}"
)

print(
    f"Frames saved     : "
    f"{saved_frames}"
)

print("\nOutput directory:")
print(
    os.path.abspath(OUTPUT_DIR)
)

print("\n" + "=" * 60)
print("PHASE 6.2 COMPLETED SUCCESSFULLY")
print("Real-time number plate detection is working.")
print("=" * 60)