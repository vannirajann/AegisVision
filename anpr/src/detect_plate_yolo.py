import cv2
import os
import tempfile
import numpy as np
from ultralytics import YOLO
from crop_plate import crop_plate, save_crop


# --------------------------------------------------
# LOAD YOLO PLATE MODEL
# --------------------------------------------------

def load_plate_model(model_path="../models/yolo_plate.pt"):
    print("Loading YOLO plate model...")

    model = YOLO(model_path)

    print("YOLO plate model loaded successfully")

    return model


# --------------------------------------------------
# DETECT NUMBER PLATES
# --------------------------------------------------

def detect_plate_yolo(model, image_input, conf_threshold=0.25):

    temp_path = None

    try:

        # If input is an OpenCV image
        if isinstance(image_input, np.ndarray):

            try:
                results = model(
                    image_input,
                    conf=conf_threshold,
                    verbose=False
                )

            except Exception as e:

                if "does not support image input" in str(e):

                    fd, temp_path = tempfile.mkstemp(
                        suffix=".jpg"
                    )

                    os.close(fd)

                    cv2.imwrite(
                        temp_path,
                        image_input
                    )

                    results = model.predict(
                        temp_path,
                        conf=conf_threshold,
                        verbose=False
                    )

                else:
                    raise

        # If input is an image path
        else:

            results = model.predict(
                image_input,
                conf=conf_threshold,
                verbose=False
            )

        detections = []

        for result in results:

            boxes = result.boxes

            if boxes is None:
                continue

            for box in boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                confidence = float(
                    box.conf[0]
                )

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": confidence
                })

        print(
            f"Found {len(detections)} plate(s)"
        )

        return detections

    finally:

        if (
            temp_path
            and os.path.exists(temp_path)
        ):
            os.unlink(temp_path)


# --------------------------------------------------
# DRAW PLATE BOXES
# --------------------------------------------------

def draw_plate_boxes(image, detections):

    output = image.copy()

    for det in detections:

        x1, y1, x2, y2 = det["bbox"]

        confidence = det["confidence"]

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        label = f"Plate {confidence:.2f}"

        cv2.putText(
            output,
            label,
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return output


# --------------------------------------------------
# TEST PLATE DETECTOR
# --------------------------------------------------

if __name__ == "__main__":

    print("-----------------------------------")
    print("ANPR YOLO Plate Detection Test")
    print("-----------------------------------")

    model = load_plate_model()

    test_files = [
        "car.jpg",
        "bike.jpg",
        "bus.jpg"
    ]

    for filename in test_files:

        path = os.path.join(
            "test_images",
            filename
        )

        print()
        print(f"Processing: {filename}")

        image = cv2.imread(path)

        if image is None:

            print(
                f"Could not load {filename}"
            )

            continue

        detections = detect_plate_yolo(
            model,
            image
        )

        result = draw_plate_boxes(
            image,
            detections
        )

        # Save detected plate crops
        for i, det in enumerate(detections):

            cropped = crop_plate(
                image,
                det["bbox"]
            )

            crop_filename = (
                f"{os.path.splitext(filename)[0]}"
                f"_plate{i}.jpg"
            )

            save_crop(
                cropped,
                crop_filename
            )

        window_name = (
            f"YOLO Plates - {filename}"
        )

        cv2.namedWindow(
            window_name,
            cv2.WINDOW_NORMAL
        )

        cv2.moveWindow(
            window_name,
            100,
            100
        )

        cv2.imshow(
            window_name,
            result
        )

        print(
            "Press any key to continue..."
        )

        cv2.waitKey(0)

        cv2.destroyWindow(
            window_name
        )

    cv2.destroyAllWindows()

    print()
    print("-----------------------------------")
    print("Plate detection test completed")
    print("-----------------------------------")