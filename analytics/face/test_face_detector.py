import cv2
import pytest
from pathlib import Path
from .face_detector import FaceDetector


def test_face_detection():

    # Test image location
    image_path = Path(__file__).parent / "test_images" / "test.jpg"

    # Check if image exists
    if not image_path.exists():
        pytest.skip(f"Test image not found: {image_path}")

    # Load image
    image = cv2.imread(str(image_path))

    assert image is not None, "Could not load test image"

    # Create detector
    detector = FaceDetector()

    # Detect faces
    faces = detector.detect_faces(image)

    print("Number of faces detected:", len(faces))

    # Draw detected faces
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

    # Save result
    output_path = Path(__file__).parent / "test_images" / "face_detection_result.jpg"

    cv2.imwrite(str(output_path), image)

    print("Result saved to:", output_path)

    # Basic test
    assert isinstance(faces, list)