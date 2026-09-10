import cv2
import os

from crop_plate import crop_plate
from preprocess_plate import preprocess_plate, preprocess_plate_gentle
from detect_plate_yolo import load_plate_model, detect_plate_yolo
from run_ocr import load_ocr_reader, run_ocr_on_plate


def best_confidence(ocr_texts):
    if not ocr_texts:
        return 0.0
    return max(t["confidence"] for t in ocr_texts)


def compare_methods(image_path, plate_model, ocr_reader):
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load {image_path}")
        return

    detections = detect_plate_yolo(plate_model, image)
    if not detections:
        print(f"No plate detected in {image_path}")
        return

    det = detections[0]  # just compare the first detected plate
    cropped = crop_plate(image, det["bbox"])

    variants = {
        "raw": cropped,
        "adaptive_threshold": preprocess_plate(cropped),
        "gentle_clahe": preprocess_plate_gentle(cropped),
    }

    print(f"\n=== {os.path.basename(image_path)} ===")
    for name, variant_image in variants.items():
        ocr_texts = run_ocr_on_plate(ocr_reader, variant_image)
        texts_joined = " | ".join(t["text"] for t in ocr_texts) if ocr_texts else "(nothing read)"
        conf = best_confidence(ocr_texts)
        print(f"  [{name:20s}] best_conf={conf:.2f}  text: {texts_joined}")


if __name__ == "__main__":
    print("Loading models...")
    plate_model = load_plate_model()
    ocr_reader = load_ocr_reader()
    print("✅ Models loaded")

    for filename in ["car.jpg", "bike.jpg", "bus.jpg"]:
        path = os.path.join("test_images", filename)
        compare_methods(path, plate_model, ocr_reader)