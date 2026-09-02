from ultralytics import YOLO


class FaceDetector:

    def __init__(self):
        print("Loading YOLO person detector...")

        # Lightweight YOLO model suitable for webcam/live detection
        self.model = YOLO("yolov8n.pt")

        # COCO dataset:
        # class 0 = person
        self.person_class_id = 0

        print("YOLO person detector loaded successfully")

    def detect_faces(self, frame):

        if frame is None:
            return []

        # Run YOLO only for the PERSON class
        results = self.model.predict(
            source=frame,
            conf=0.45,
            iou=0.50,
            classes=[self.person_class_id],
            imgsz=640,
            verbose=False
        )

        detections = []

        if not results:
            return detections

        result = results[0]

        if result.boxes is None:
            return detections

        # Process every detected person
        for box in result.boxes:

            confidence = float(box.conf[0])

            # Ignore low-confidence detections
            if confidence < 0.45:
                continue

            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            width = x2 - x1
            height = y2 - y1

            if width <= 0 or height <= 0:
                continue

            detections.append({
                "x": x1,
                "y": y1,
                "width": width,
                "height": height,
                "confidence": round(confidence, 3),
                "class": "person"
            })

        return detections

    def detect(self, frame):
        return self.detect_faces(frame)