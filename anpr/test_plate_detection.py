import cv2
from ultralytics import YOLO
import os

MODEL_PATH = "models/plate_detector.pt"

print("=" * 70)
print("AEGISVISION - ANPR")
print("PHASE 6.6 - YOLO PLATE DETECTION TEST")
print("=" * 70)

# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

print("\nLoading YOLO model...")

if not os.path.exists(MODEL_PATH):
    print("ERROR: Model not found:")
    print(MODEL_PATH)
    raise SystemExit

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully.")

# ---------------------------------------------------------
# Open webcam
# ---------------------------------------------------------

print("\nOpening webcam...")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    raise SystemExit

print("Camera opened successfully.")

print("\nControls:")
print("Q / ESC = Quit")
print("S       = Save detected plate crop")

# ---------------------------------------------------------
# Detection settings
# ---------------------------------------------------------

CONFIDENCE = 0.15
IMAGE_SIZE = 640

frame_count = 0
detection_count = 0

os.makedirs("output/phase6_6", exist_ok=True)

# ---------------------------------------------------------
# Main loop
# ---------------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    frame_count += 1

    # YOLO detection
    results = model.predict(
        source=frame,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        verbose=False
    )

    result = results[0]

    boxes = result.boxes

    if boxes is not None and len(boxes) > 0:

        for i, box in enumerate(boxes):

            detection_count += 1

            xyxy = box.xyxy[0].cpu().numpy().astype(int)

            x1, y1, x2, y2 = xyxy

            confidence = float(box.conf[0])

            # Keep coordinates inside image
            h, w = frame.shape[:2]

            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            # Crop plate
            plate_crop = frame[y1:y2, x1:x2]

            # Draw detection
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = f"PLATE {confidence * 100:.1f}%"

            cv2.putText(
                frame,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # Save latest crop
            crop_path = "output/phase6_6/detected_plate.jpg"

            cv2.imwrite(crop_path, plate_crop)

            print(
                f"[DETECTION] "
                f"confidence={confidence * 100:.2f}% "
                f"box=({x1},{y1},{x2},{y2})"
            )

    # Display information
    cv2.putText(
        frame,
        f"Frames: {frame_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Detections: {detection_count}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("AEGISVISION - Plate Detection Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break

    if key == ord("s"):
        cv2.imwrite(
            "output/phase6_6/test_frame.jpg",
            frame
        )
        print("[SAVED] Test frame saved.")


# ---------------------------------------------------------
# Cleanup
# ---------------------------------------------------------

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 70)
print("PHASE 6.6 YOLO TEST RESULT")
print("=" * 70)

print(f"Frames processed : {frame_count}")
print(f"Detections       : {detection_count}")

print("\nOutput:")
print("output/phase6_6/detected_plate.jpg")
print("output/phase6_6/test_frame.jpg")

print("=" * 70)