import cv2
from movement_detector import MovementDetector


# Create the movement detector
detector = MovementDetector(threshold=25)

# Load the first image
frame1 = cv2.imread("test_images/frame1.jpg")

# Load the second image
frame2 = cv2.imread("test_images/frame2.jpg")


# Check that the images were loaded
if frame1 is None:
    print("ERROR: frame1.jpg could not be loaded")
    exit()

if frame2 is None:
    print("ERROR: frame2.jpg could not be loaded")
    exit()


print("Frame 1 loaded successfully")
print("Frame 2 loaded successfully")


# Use the first image as the reference
detector.detect_movement(frame1)

# Compare the second image with the first image
movement_detected = detector.detect_movement(frame2)


if movement_detected:
    print("Movement detected!")
else:
    print("No significant movement detected.")