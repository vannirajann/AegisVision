import os
import sys

# Add analytics folder to Python path
analytics_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, analytics_root)

from face.face_detector import FaceDetector


def test_face_detector_creation():
    """Test that FaceDetector can be created without opening the camera."""

    detector = FaceDetector()

    assert detector is not None

    print("Face detector created successfully")