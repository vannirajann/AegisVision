import cv2


class MovementDetector:
    def __init__(self, threshold=3.0):
        self.threshold = threshold
        self.previous_frame = None

    def get_change_amount(self, frame):
        """
        Fast movement detection using a small grayscale image.
        """

        # Convert directly to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Very small image = much faster processing
        gray = cv2.resize(gray, (160, 120))

        # Compare with previous frame
        if self.previous_frame is None:
            self.previous_frame = gray
            return 0.0

        difference = cv2.absdiff(
            self.previous_frame,
            gray
        )

        # Average difference
        change_amount = float(difference.mean())

        # Save current frame
        self.previous_frame = gray

        return change_amount

    def detect_movement(self, frame):
        change_amount = self.get_change_amount(frame)

        return change_amount > self.threshold