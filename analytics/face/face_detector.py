import cv2


class FaceDetector:
    def __init__(self):
        # Load OpenCV's built-in Haar Cascade face detector
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            raise RuntimeError("Could not load face detection model.")

    def detect_faces(self, frame):
        """
        Detect faces in an image/frame.

        Returns:
            List of dictionaries containing face bounding boxes.
        """

        # Convert the image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        results = []

        for (x, y, width, height) in faces:
            results.append({
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height)
            })

        return results