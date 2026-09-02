import cv2


class NightDetector:

    def __init__(self, brightness_threshold=80):
        """
        Detect whether a video frame is low-light/night.

        brightness_threshold:
            Average brightness below this value
            is considered low-light.
        """

        self.brightness_threshold = brightness_threshold

    def get_brightness(self, frame):
        """
        Calculate the average brightness of a frame.
        """

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        brightness = gray.mean()

        return float(brightness)

    def is_low_light(self, frame):
        """
        Return True if the frame is considered low-light.
        """

        brightness = self.get_brightness(frame)

        return brightness < self.brightness_threshold

    def detect(self, frame):
        """
        Return complete low-light detection information.
        """

        brightness = self.get_brightness(frame)

        low_light = brightness < self.brightness_threshold

        if low_light:
            status = "LOW LIGHT / NIGHT"
        else:
            status = "NORMAL LIGHT"

        return {
            "low_light": low_light,
            "brightness": round(brightness, 2),
            "status": status
        }