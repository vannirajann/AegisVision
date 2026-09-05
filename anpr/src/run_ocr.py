import cv2
import os
import easyocr

def load_ocr_reader():
    # 'en' = English. This also handles most alphanumeric plates fine.
    # gpu=False since we're not assuming a GPU is available.
    reader = easyocr.Reader(['en'], gpu=False)
    return reader

def run_ocr_on_plate(reader, cropped_image):
    results = reader.readtext(
        cropped_image,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    texts = []
    for (bbox, text, confidence) in results:
        texts.append({"text": text, "confidence": confidence})

    return texts

if __name__ == "__main__":
    reader = load_ocr_reader()

    crop_files = ["car_plate0.jpg", "bike_plate0.jpg", "bus_plate0.jpg"]

    for filename in crop_files:
        path = os.path.join("output_crops", filename)
        image = cv2.imread(path)

        if image is None:
            print(f"❌ Could not load {filename}")
            continue

        print(f"--- {filename} ---")
        texts = run_ocr_on_plate(reader, image)

        if not texts:
            print("  ⚠️ No text detected")
        for t in texts:
            print(f"  📝 '{t['text']}'  (confidence: {t['confidence']:.2f})")