import cv2


class MovementDetector:
    def __init__(self, threshold=25):
        self.threshold = threshold
        self.previous_frame = None

    def prepare_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        return gray

    def detect_movement(self, frame):
        gray = self.prepare_frame(frame)

        # First frame becomes the reference
        if self.previous_frame is None:
            self.previous_frame = gray
            return False

        # Resize current frame to exactly match previous frame
        gray = cv2.resize(
            gray,
            (
                self.previous_frame.shape[1],
                self.previous_frame.shape[0]
            )
        )

        # Compare the two frames
        frame_difference = cv2.absdiff(
            self.previous_frame,
            gray
        )

        # Calculate how much the image changed
        change_amount = frame_difference.mean()

        # Save current frame for next comparison
        self.previous_frame = gray

        return change_amount > self.threshold

    def get_change_amount(self, frame):
        gray = self.prepare_frame(frame)

        if self.previous_frame is None:
            self.previous_frame = gray
            return 0.0

        # Always make the dimensions identical
        gray = cv2.resize(
            gray,
            (
                self.previous_frame.shape[1],
                self.previous_frame.shape[0]
            )
        )

        frame_difference = cv2.absdiff(
            self.previous_frame,
            gray
        )

        change_amount = frame_difference.mean()

        self.previous_frame = gray

        return float(change_amount)