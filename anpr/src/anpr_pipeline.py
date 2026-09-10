import cv2
import os
import json

from preprocess_plate import preprocess_plate, preprocess_plate_gentle
from detect_plate_yolo import load_plate_model, detect_plate_yolo
from crop_plate import crop_plate
from run_ocr import load_ocr_reader, run_ocr_on_plate
from clean_text import clean_ocr_fragments
from build_output import build_anpr_result


def draw_result_on_image(image, det, result):
    x1, y1, x2, y2 = det["bbox"]
    color = (0, 255, 0) if result["plate_valid_format"] else (0, 165, 255)
    output = image.copy()
    cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
    label = f"{result['plate_number']} ({result['confidence']:.2f})"
    cv2.putText(output, label, (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return output


def save_image_result(image_path, image, results, detections, output_base_dir):
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    annotated_dir = os.path.join(output_base_dir, "annotated")
    results_dir = os.path.join(output_base_dir, "results")
    os.makedirs(annotated_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Draw all detections on one image
    annotated = image.copy()
    for det, result in zip(detections, results):
        annotated = draw_result_on_image(annotated, det, result)

    annotated_path = os.path.join(annotated_dir, f"{base_name}_result.jpg")
    cv2.imwrite(annotated_path, annotated)
    print(f"💾 Saved annotated image: {annotated_path}")

    results_path = os.path.join(results_dir, f"{base_name}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"💾 Saved results JSON: {results_path}")


def run_anpr_pipeline(image_path, plate_model, ocr_reader):
    """
    Full ANPR pipeline: image path -> structured JSON result(s).
    Returns (results, detections) - a list of results and their matching
    detections (one per detected plate in the image).
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return [], []

    detections = detect_plate_yolo(plate_model, image)
    results = []

    for det in detections:
        cropped = crop_plate(image, det["bbox"])
        if cropped is None:
            continue

        # Try both preprocessing methods, keep whichever gives a better cleaned result
        variant_a = preprocess_plate(cropped)
        variant_b = preprocess_plate_gentle(cropped)

        ocr_texts_a = run_ocr_on_plate(ocr_reader, variant_a)
        ocr_texts_b = run_ocr_on_plate(ocr_reader, variant_b)

        plate_text_a, valid_a = clean_ocr_fragments(ocr_texts_a)
        plate_text_b, valid_b = clean_ocr_fragments(ocr_texts_b)

        # Prefer a valid-format result; if both/neither valid, prefer the longer cleaned text
        if valid_a and not valid_b:
            plate_text, is_valid, ocr_texts = plate_text_a, valid_a, ocr_texts_a
        elif valid_b and not valid_a:
            plate_text, is_valid, ocr_texts = plate_text_b, valid_b, ocr_texts_b
        elif len(plate_text_a) >= len(plate_text_b):
            plate_text, is_valid, ocr_texts = plate_text_a, valid_a, ocr_texts_a
        else:
            plate_text, is_valid, ocr_texts = plate_text_b, valid_b, ocr_texts_b

        # Average confidence across kept OCR fragments (fallback to detection conf if empty)
        if ocr_texts:
            avg_conf = sum(t["confidence"] for t in ocr_texts) / len(ocr_texts)
        else:
            avg_conf = det["confidence"]

        result = build_anpr_result(plate_text, is_valid, avg_conf)
        results.append(result)

    return results, detections


if __name__ == "__main__":
    print("Loading models... (this happens once)")
    plate_model = load_plate_model()
    ocr_reader = load_ocr_reader()
    print("✅ Models loaded\n")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_images_dir = os.path.join(base_dir, "..", "test_images")
    output_base_dir = os.path.join(base_dir, "..", "output")

    test_files = ["car.jpeg", "car2.jpeg", "car3.jpeg"]

    for filename in test_files:
        path = os.path.join(test_images_dir, filename)
        print(f"--- Processing {filename} ---")

        results, detections = run_anpr_pipeline(path, plate_model, ocr_reader)

        for r in results:
            print(json.dumps(r, indent=2))

        if results:
            image = cv2.imread(path)
            save_image_result(path, image, results, detections, output_base_dir)
        print()