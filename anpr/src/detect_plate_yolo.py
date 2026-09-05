import cv2
import os
import tempfile
import numpy as np
from ultralytics import YOLO
from crop_plate import crop_plate, save_crop

def load_plate_model(model_path="models/yolo_plate.pt"):
    model = YOLO(model_path)
    return model

def detect_plate_yolo(model, image_input, conf_threshold=0.25):
    """
    Run plate detection on an image.
    Accepts file path (str) or numpy array (OpenCV frame).
    Falls back to temp file if model does not support direct image input.
    """
    temp_path = None
    try:
        if isinstance(image_input, np.ndarray):
            try:
                results = model(image_input, conf=conf_threshold, verbose=False)
            except Exception as e:
                if "does not support image input" in str(e):
                    fd, temp_path = tempfile.mkstemp(suffix=".jpg")
                    os.close(fd)
                    cv2.imwrite(temp_path, image_input)
                    results = model.predict(temp_path, conf=conf_threshold, verbose=False)
                else:
                    raise
        else:
            results = model.predict(image_input, conf=conf_threshold, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
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