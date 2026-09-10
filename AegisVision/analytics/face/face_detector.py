import cv2


class FaceDetector:
    def __init__(self):
        # Load OpenCV's built-in Haar Cascade face detector
        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            raise RuntimeError(
                "Could not load face detection model."
            )

    def detect_faces(self, frame):
        """
        Detect real faces in an image/frame.

        Returns:
            List of dictionaries containing face bounding boxes.
        """

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Improve contrast
        gray = cv2.equalizeHist(gray)

        # Detect faces with stricter settings
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=8,
            minSize=(60, 60),
            maxSize=(400, 400)
        )

        results = []

        for (x, y, width, height) in faces:

            # Reject unusually small detections
            if width < 60 or height < 60:
                continue

            # A real face should roughly have a square/portrait shape
            aspect_ratio = width / float(height)

            if aspect_ratio < 0.65 or aspect_ratio > 1.35:
                continue

            results.append({
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height)
            })

        return results