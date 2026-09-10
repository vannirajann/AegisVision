import cv2
import os
import tempfile
import numpy as np
from ultralytics import YOLO
from crop_plate import crop_plate, save_crop


def load_plate_model(model_path=None):
    """
    Loads the YOLO plate-detection model.
    If no path is given, resolves it relative to this file's own location
    (../models/yolo_plate.pt) so it works no matter which directory
    the script is run from.
    """
    if model_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "..", "models", "yolo_plate.pt")
    model = YOLO(model_path)
    return model


def detect_plate_yolo(model, image_input, conf_threshold=0.45, iou_threshold=0.4,
                       min_box_area=400, min_aspect_ratio=1.5, max_aspect_ratio=6.0):
    """
    Run plate detection on an image.
    Accepts file path (str) or numpy array (OpenCV frame).
    Falls back to temp file if model does not support direct image input.

    conf_threshold raised from 0.25 -> 0.45 to reject weak/false detections
    (grilles, headlights, bumper trim being mistaken for plates).
    iou_threshold explicitly set to prevent duplicate overlapping boxes on
    the same vehicle.
    min_box_area / aspect ratio filters reject boxes that are too small or
    the wrong shape to plausibly be a real plate.
    """
    temp_path = None
    try:
        if isinstance(image_input, np.ndarray):
            try:
                results = model(image_input, conf=conf_threshold, iou=iou_threshold, verbose=False)
            except Exception as e:
                if "does not support image input" in str(e):
                    fd, temp_path = tempfile.mkstemp(suffix=".jpg")
                    os.close(fd)
                    cv2.imwrite(temp_path, image_input)
                    results = model.predict(temp_path, conf=conf_threshold, iou=iou_threshold, verbose=False)
                else:
                    raise
        else:
            results = model.predict(image_input, conf=conf_threshold, iou=iou_threshold, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                width = x2 - x1
                height = y2 - y1
                area = width * height
                aspect_ratio = width / height if height > 0 else 0

                # Reject boxes too small or the wrong shape to be a real plate
                if area < min_box_area:
                    continue
                if not (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio):
                    continue

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": confidence
                })

        print(f"🔍 Found {len(detections)} plate(s)")
        return detections
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def draw_plate_boxes(image, detections):
    output = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"plate {conf:.2f}"
        cv2.putText(output, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return output


if __name__ == "__main__":
    model = load_plate_model()

    test_files = ["car.jpg", "bike.jpg", "bus.jpg"]

    for filename in test_files:
        path = os.path.join("test_images", filename)
        image = cv2.imread(path)

        if image is None:
            print(f"❌ Could not load {filename}")
            continue

        print(f"--- {filename} ---")
        detections = detect_plate_yolo(model, image)
        result = draw_plate_boxes(image, detections)

        for i, det in enumerate(detections):
            cropped = crop_plate(image, det["bbox"])
            if cropped is None:
                continue
            crop_filename = f"{os.path.splitext(filename)[0]}_plate{i}.jpg"
            save_crop(cropped, crop_filename)

        window_name = f"YOLO Plates - {filename}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, 100, 100)
        cv2.imshow(window_name, result)
        print(">>> Click the window, press any key to continue <<<")
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)

    cv2.destroyAllWindows()