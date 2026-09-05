import os
import sys

# Add analytics folder to Python path
analytics_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, analytics_root)

from video.video_source import VideoSource


def test_video_source_creation():
    """Test that VideoSource can be created."""

    source = VideoSource(0)

    assert source is not None

    print("VideoSource created successfully")