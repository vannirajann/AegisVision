import cv2
from movement_detector import MovementDetector

print("Starting movement detection test...")

# Create movement detector
detector = MovementDetector(threshold=25)

# Load first image
frame1 = cv2.imread("test_images/frame1.jpg")

# Load second image
frame2 = cv2.imread("test_images/frame2.jpg")

# Check images
if frame1 is None or frame2 is None:
    print("Test images not found.")
    print("Skipping movement detection test.")
else:
    print("Frame 1 loaded successfully")
    print("Frame 2 loaded successfully")

    # Set first frame as reference
    detector.detect_movement(frame1)

    # Compare second frame
    movement_detected = detector.detect_movement(frame2)

    if movement_detected:
        print("Movement detected!")
    else:
        print("No significant movement detected.")

print("Movement detection test completed.")