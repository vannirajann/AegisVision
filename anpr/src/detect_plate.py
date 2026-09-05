import cv2
import os

def detect_plate(image, cascade_path="models/haarcascade_russian_plate_number.xml"):
    # Load the cascade classifier
    plate_cascade = cv2.CascadeClassifier(cascade_path)

    if plate_cascade.empty():
        print("❌ Could not load cascade file. Check the path.")
        return []

    # Haar cascades work on grayscale images
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect plates - returns a list of (x, y, w, h) boxes
    plates = plate_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 20)
    )

    print(f"🔍 Found {len(plates)} potential plate(s)")
    return plates

def draw_plates(image, plates):
    output = image.copy()
    for (x, y, w, h) in plates:
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return output

if __name__ == "__main__":
    test_files = ["car.jpg", "bike.jpg", "bus.jpg"]

    for filename in test_files:
        path = os.path.join("test_images", filename)
        image = cv2.imread(path)

        if image is None:
            print(f"❌ Could not load {filename}")
            continue

        plates = detect_plate(image)
        result = draw_plates(image, plates)

        print(f"--- {filename}: {len(plates)} plate(s) found ---")

        window_name = f"Detected Plates - {filename}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, 100, 100)
        cv2.imshow(window_name, result)
        print(">>> Click on the image window, then press any key to continue <<<")
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)

    cv2.destroyAllWindows()