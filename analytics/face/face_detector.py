import os
import cv2
import urllib.request


class FaceDetector:

    def __init__(self):

        print("Loading YuNet face detector...")

        # ==========================================
        # Model location
        # ==========================================

        BASE_DIR = os.path.dirname(
            os.path.abspath(__file__)
        )

        MODEL_DIR = os.path.join(
            BASE_DIR,
            "models"
        )

        os.makedirs(
            MODEL_DIR,
            exist_ok=True
        )

        self.model_path = os.path.join(
            MODEL_DIR,
            "face_detection_yunet_2023mar.onnx"
        )

        # ==========================================
        # YuNet model URL
        # ==========================================

        MODEL_URL = (
            "https://github.com/opencv/opencv_zoo/"
            "raw/main/models/face_detection_yunet/"
            "face_detection_yunet_2023mar.onnx"
        )

        # ==========================================
        # Download model if it doesn't exist
        # ==========================================

        if not os.path.exists(self.model_path):

            print("YuNet model not found.")
            print("Downloading YuNet face detection model...")

            try:

                urllib.request.urlretrieve(
                    MODEL_URL,
                    self.model_path
                )

                print(
                    "YuNet model downloaded successfully."
                )

            except Exception as e:

                print(
                    "ERROR: Could not download YuNet model."
                )

                print(e)

                raise

        # ==========================================
        # Face detector settings
        # ==========================================

        self.confidence_threshold = 0.75

        self.nms_threshold = 0.3

        self.top_k = 5000

        # ==========================================
        # Create YuNet detector
        # ==========================================

        self.detector = cv2.FaceDetectorYN.create(
            self.model_path,
            "",
            (320, 320),
            self.confidence_threshold,
            self.nms_threshold,
            self.top_k
        )

        print(
            "YuNet face detector loaded successfully"
        )


    def detect_faces(self, frame):

        if frame is None:
            return []

        if frame.size == 0:
            return []

        # ==========================================
        # Get frame dimensions
        # ==========================================

        height, width = frame.shape[:2]

        # ==========================================
        # Tell YuNet the current image size
        # ==========================================

        self.detector.setInputSize(
            (width, height)
        )

        # ==========================================
        # Detect ONLY faces
        # ==========================================

        _, detections = self.detector.detect(
            frame
        )

        faces = []

        if detections is None:
            return faces

        # ==========================================
        # Process detections
        # ==========================================

        for detection in detections:

            # YuNet format:
            #
            # 0 = x
            # 1 = y
            # 2 = width
            # 3 = height
            #
            # 4-13 = facial landmarks
            #
            # 14 = confidence
            # ==========================================

            x = int(detection[0])
            y = int(detection[1])
            w = int(detection[2])
            h = int(detection[3])

            confidence = float(
                detection[14]
            )

            # ==========================================
            # Extra confidence filtering
            # ==========================================

            if confidence < self.confidence_threshold:
                continue

            # ==========================================
            # Remove invalid boxes
            # ==========================================

            if w <= 0 or h <= 0:
                continue

            # ==========================================
            # Keep box inside image
            # ==========================================

            x = max(0, x)
            y = max(0, y)

            w = min(
                w,
                width - x
            )

            h = min(
                h,
                height - y
            )

            if w <= 0 or h <= 0:
                continue

            # ==========================================
            # Add real face detection
            # ==========================================

            faces.append(
                {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "confidence": round(
                        confidence,
                        3
                    )
                }
            )

        return faces