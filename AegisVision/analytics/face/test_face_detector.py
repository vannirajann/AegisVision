import cv2
from face_detector import FaceDetector


# Path to our test image
IMAGE_PATH = "test_images/test.jpg"

# Load the image
image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

# Create the face detector
detector = FaceDetector()

# Detect faces
faces = detector.detect_faces(image)

print("Number of faces detected:", len(faces))

# Draw bounding boxes around detected faces
for face in faces:
    x = face["x"]
    y = face["y"]
    width = face["width"]
    height = face["height"]

    cv2.rectangle(
        image,
        (x, y),
        (x + width, y + height),
        (0, 255, 0),
        2
    )

    print(
        "Face:",
        f"x={x}, y={y}, width={width}, height={height}"
    )

# Save the result
OUTPUT_PATH = "test_images/face_detection_result.jpg"
cv2.imwrite(OUTPUT_PATH, image)

print("Result saved to:", OUTPUT_PATH)