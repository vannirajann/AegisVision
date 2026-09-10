import cv2


class VideoSource:
    def __init__(self, source=0):
        """
        Create a video source.

        source=0 means the default webcam.
        """

        self.source = source
        self.cap = None

    def open(self):
        """Open the camera or video source."""

        self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open video source")

        return True

    def read(self):
        """Read one frame from the video source."""

        if self.cap is None:
            return False, None

        return self.cap.read()

    def release(self):
        """Release the video source."""

        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def is_opened(self):
        """Check whether the video source is open."""

        return self.cap is not None and self.cap.isOpened()