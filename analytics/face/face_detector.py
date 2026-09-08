import cv2
import os
import math
from ultralytics import YOLO


class FaceDetector:

    def __init__(self):

        print("Loading YOLO person detector...")

        # --------------------------------------------------
        # PROJECT PATHS
        # --------------------------------------------------

        analytics_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                ".."
            )
        )

        project_dir = os.path.abspath(
            os.path.join(
                analytics_dir,
                ".."
            )
        )

        yolo_path = os.path.join(
            project_dir,
            "detection",
            "yolo11n.pt"
        )

        # --------------------------------------------------
        # YOLO PERSON DETECTOR
        # --------------------------------------------------

        self.person_model = YOLO(yolo_path)

        print("YOLO person detector loaded successfully")

        # --------------------------------------------------
        # FIND YUNET MODEL
        # --------------------------------------------------

        yunet_model = None

        search_folders = [
            os.path.dirname(__file__),
            analytics_dir,
            project_dir
        ]

        for folder in search_folders:

            if not os.path.exists(folder):
                continue

            for root, dirs, files in os.walk(folder):

                for filename in files:

                    if (
                        filename.lower().endswith(".onnx")
                        and "yunet" in filename.lower()
                    ):
                        yunet_model = os.path.join(
                            root,
                            filename
                        )
                        break

                if yunet_model:
                    break

            if yunet_model:
                break

        if yunet_model is None:

            raise FileNotFoundError(
                "YuNet ONNX model not found."
            )

        print("YuNet model found:")
        print(yunet_model)

        # --------------------------------------------------
        # YUNET
        # --------------------------------------------------

        self.yunet = cv2.FaceDetectorYN.create(
            yunet_model,
            "",
            (320, 320),

            # Face confidence
            0.45,

            # NMS threshold
            0.30,

            # Maximum detections
            5000
        )

        print("YuNet face detector loaded successfully")

    # ======================================================
    # CHECK FACIAL LANDMARKS
    # ======================================================

    def valid_face_landmarks(self, face):

        try:

            # YuNet landmark positions
            #
            # 0-1   = right eye
            # 2-3   = left eye
            # 4-5   = nose
            # 6-7   = right mouth
            # 8-9   = left mouth

            right_eye_x = float(face[4])
            right_eye_y = float(face[5])

            left_eye_x = float(face[6])
            left_eye_y = float(face[7])

            nose_x = float(face[8])
            nose_y = float(face[9])

            face_x = float(face[0])
            face_y = float(face[1])
            face_w = float(face[2])
            face_h = float(face[3])

            if face_w <= 0 or face_h <= 0:
                return False

            # --------------------------------------------------
            # EYES MUST BE INSIDE FACE
            # --------------------------------------------------

            if not (
                face_x <= right_eye_x <= face_x + face_w
                and
                face_y <= right_eye_y <= face_y + face_h
            ):
                return False

            if not (
                face_x <= left_eye_x <= face_x + face_w
                and
                face_y <= left_eye_y <= face_y + face_h
            ):
                return False

            # --------------------------------------------------
            # NOSE MUST BE INSIDE FACE
            # --------------------------------------------------

            if not (
                face_x <= nose_x <= face_x + face_w
                and
                face_y <= nose_y <= face_y + face_h
            ):
                return False

            # --------------------------------------------------
            # EYES SHOULD BE SEPARATED
            # --------------------------------------------------

            eye_distance = math.sqrt(
                (right_eye_x - left_eye_x) ** 2
                +
                (right_eye_y - left_eye_y) ** 2
            )

            # If the two eyes are almost at the same point,
            # it is probably a false detection.

            if eye_distance < face_w * 0.18:
                return False

            # --------------------------------------------------
            # EYES SHOULD BE IN UPPER HALF
            # --------------------------------------------------

            average_eye_y = (
                right_eye_y + left_eye_y
            ) / 2

            relative_eye_y = (
                average_eye_y - face_y
            ) / face_h

            if relative_eye_y > 0.65:
                return False

            # --------------------------------------------------
            # NOSE SHOULD BE BELOW EYES
            # --------------------------------------------------

            if nose_y < average_eye_y:
                return False

            return True

        except Exception:
            return False

    # ======================================================
    # CHECK FACE SHAPE
    # ======================================================

    def valid_face_shape(self, face):

        face_w = float(face[2])
        face_h = float(face[3])

        if face_w <= 0 or face_h <= 0:
            return False

        ratio = face_w / face_h

        # Human faces are normally approximately rectangular.
        # This removes very strange detections.

        if ratio < 0.55:
            return False

        if ratio > 1.45:
            return False

        return True

    # ======================================================
    # DETECT FACES
    # ======================================================

    def detect_faces(self, frame):

        if frame is None:
            return []

        detected_faces = []

        frame_height, frame_width = frame.shape[:2]

        # --------------------------------------------------
        # YOLO PERSON DETECTION
        # --------------------------------------------------

        results = self.person_model(
            frame,

            # COCO class 0 = person
            classes=[0],

            # Person confidence
            conf=0.30,

            verbose=False
        )

        if not results:
            return []

        boxes = results[0].boxes

        if boxes is None:
            return []

        # --------------------------------------------------
        # PROCESS EACH PERSON
        # --------------------------------------------------

        for box in boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # Keep coordinates inside frame

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(
                frame_width,
                x2
            )

            y2 = min(
                frame_height,
                y2
            )

            person_width = x2 - x1
            person_height = y2 - y1

            # --------------------------------------------------
            # IGNORE VERY SMALL PERSONS
            # --------------------------------------------------

            if person_width < 25:
                continue

            if person_height < 70:
                continue

            # --------------------------------------------------
            # FACE SEARCH REGION
            #
            # We search only the upper part of the person.
            # This prevents detecting objects around the person.
            # --------------------------------------------------

            head_y1 = y1

            head_y2 = y1 + int(
                person_height * 0.55
            )

            person_crop = frame[
                head_y1:head_y2,
                x1:x2
            ]

            if person_crop.size == 0:
                continue

            crop_height, crop_width = person_crop.shape[:2]

            if crop_width < 20 or crop_height < 20:
                continue

            # --------------------------------------------------
            # ENLARGE PERSON REGION
            # --------------------------------------------------

            scale = 2.5

            enlarged = cv2.resize(
                person_crop,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

            new_height, new_width = enlarged.shape[:2]

            # --------------------------------------------------
            # YUNET INPUT SIZE
            # --------------------------------------------------

            self.yunet.setInputSize(
                (new_width, new_height)
            )

            # --------------------------------------------------
            # YUNET DETECTION
            # --------------------------------------------------

            _, faces = self.yunet.detect(
                enlarged
            )

            if faces is None:
                continue

            # --------------------------------------------------
            # CHECK EACH YUNET FACE
            # --------------------------------------------------

            for face in faces:

                confidence = float(
                    face[14]
                )

                # --------------------------------------------------
                # HIGHER CONFIDENCE FILTER
                # --------------------------------------------------

                if confidence < 0.45:
                    continue

                # --------------------------------------------------
                # FACE SHAPE FILTER
                # --------------------------------------------------

                if not self.valid_face_shape(face):
                    continue

                # --------------------------------------------------
                # FACIAL LANDMARK FILTER
                # --------------------------------------------------

                if not self.valid_face_landmarks(face):
                    continue

                # --------------------------------------------------
                # CONVERT YUNET COORDINATES
                # --------------------------------------------------

                fx = float(face[0]) / scale
                fy = float(face[1]) / scale

                fw = float(face[2]) / scale
                fh = float(face[3]) / scale

                final_x = int(
                    x1 + fx
                )

                final_y = int(
                    head_y1 + fy
                )

                final_w = int(fw)
                final_h = int(fh)

                # --------------------------------------------------
                # FACE MUST BE INSIDE PERSON
                # --------------------------------------------------

                if final_x < x1:
                    continue

                if final_y < y1:
                    continue

                if final_x + final_w > x2:
                    continue

                if final_y + final_h > head_y2:
                    continue

                # --------------------------------------------------
                # FACE SIZE FILTER
                # --------------------------------------------------

                if final_w < 12:
                    continue

                if final_h < 12:
                    continue

                # Face should not be absurdly large compared
                # with the person.

                if final_w > person_width * 0.85:
                    continue

                if final_h > person_height * 0.50:
                    continue

                # --------------------------------------------------
                # FACE CENTER FILTER
                # --------------------------------------------------

                face_center_x = (
                    final_x + final_w / 2
                )

                person_center_x = (
                    x1 + person_width / 2
                )

                horizontal_difference = abs(
                    face_center_x - person_center_x
                )

                # Allow some side-facing people.
                # But reject detections very far outside
                # the person's central head area.

                if horizontal_difference > person_width * 0.48:
                    continue

                # --------------------------------------------------
                # SAVE VALID FACE
                # --------------------------------------------------

                detected_faces.append({
                    "x": final_x,
                    "y": final_y,
                    "width": final_w,
                    "height": final_h,
                    "confidence": round(
                        confidence,
                        2
                    )
                })

        # --------------------------------------------------
        # REMOVE DUPLICATE FACES
        # --------------------------------------------------

        final_faces = []

        for face in detected_faces:

            duplicate = False

            cx1 = (
                face["x"]
                + face["width"] / 2
            )

            cy1 = (
                face["y"]
                + face["height"] / 2
            )

            for existing in final_faces:

                cx2 = (
                    existing["x"]
                    + existing["width"] / 2
                )

                cy2 = (
                    existing["y"]
                    + existing["height"] / 2
                )

                distance = math.sqrt(
                    (cx1 - cx2) ** 2
                    +
                    (cy1 - cy2) ** 2
                )

                if distance < min(
                    face["width"],
                    face["height"]
                ) * 0.5:

                    duplicate = True
                    break

            if not duplicate:
                final_faces.append(face)

        return final_faces