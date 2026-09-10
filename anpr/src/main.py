import os
import numpy as np
from ultralytics import YOLO
import cv2

from sort import Sort
from util import get_car, read_license_plate, write_csv
from run_ocr import load_ocr_reader

results = {}
mot_tracker = Sort(max_age=15, min_hits=1, iou_threshold=0.2)

base_dir = os.path.dirname(os.path.abspath(__file__))
plate_model_path = os.path.join(base_dir, "..", "models", "yolo_plate.pt")
video_path = os.path.join(base_dir, "..", "test_images", "test_video.mp4")
output_csv = os.path.join(base_dir, "..", "output", "results", "test.csv")

os.makedirs(os.path.dirname(output_csv), exist_ok=True)

coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO(plate_model_path)
ocr_reader = load_ocr_reader()

cap = cv2.VideoCapture(video_path)
vehicles = [2, 3, 5, 7]  # car, motorcycle, bus, truck

frame_nmr = -1
ret = True

while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        if frame_nmr % 30 == 0:
            print(f"Processing frame {frame_nmr}...")

        results[frame_nmr] = {}

        detections = coco_model(frame, conf=0.3, verbose=False)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        track_ids = mot_tracker.update(np.asarray(detections_) if detections_ else np.empty((0, 5)))

        license_plates = license_plate_detector(frame, conf=0.2, verbose=False)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                if license_plate_crop.size == 0:
                    continue

                license_plate_text, license_plate_text_score = read_license_plate(
                    license_plate_crop, ocr_reader
                )

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'text': license_plate_text,
                            'bbox_score': score,
                            'text_score': license_plate_text_score
                        }
                    }

write_csv(results, output_csv)
print(f"✅ Done. Results saved to {output_csv}")