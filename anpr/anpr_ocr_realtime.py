import os

# ============================================================
# IMPORTANT:
# Disable oneDNN BEFORE importing Paddle/PaddleOCR.
# This avoids the ConvertPirAttribute2RuntimeAttribute error.
# ============================================================

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

import cv2
import re
import time
import json

from ultralytics import YOLO
from paddleocr import PaddleOCR


# ============================================================
# AEGISVISION - ANPR
# PHASE 6.2 - REAL-TIME OCR INTEGRATION
# YOLO + PaddleOCR + WEBCAM
# ============================================================

print("=" * 60)
print("AEGISVISION - ANPR")
print("PHASE 6.2 - REAL-TIME OCR INTEGRATION")
print("YOLO + PaddleOCR + WEBCAM")
print("=" * 60)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "plate_detector.pt"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output",
    "phase6_2"
)

PLATE_CROP_DIR = os.path.join(
    OUTPUT_DIR,
    "plate_crops"
)

OCR_HISTORY_FILE = os.path.join(
    OUTPUT_DIR,
    "ocr_history.json"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLATE_CROP_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

CAMERA_INDEX = 0

CONFIDENCE_THRESHOLD = 0.25

MIN_PLATE_WIDTH = 40
MIN_PLATE_HEIGHT = 15

# Run OCR every N frames
OCR_INTERVAL = 10

PLATE_PADDING = 5

# Minimum OCR confidence
MIN_OCR_CONFIDENCE = 0.30

WINDOW_NAME = "AegisVision - Real-Time ANPR"


# ============================================================
# CHECK YOLO MODEL
# ============================================================

print()
print("Loading YOLO plate detector...")

if not os.path.exists(MODEL_PATH):

    print()
    print("ERROR: YOLO model not found!")
    print()
    print("Expected:")
    print(MODEL_PATH)
    print()
    print("Make sure plate_detector.pt exists inside:")
    print("anpr/models/")
    print()

    raise SystemExit(1)


try:

    detector = YOLO(MODEL_PATH)

    print("YOLO detector loaded successfully.")

except Exception as e:

    print("ERROR loading YOLO model:")
    print(e)

    raise SystemExit(1)


# ============================================================
# LOAD PADDLEOCR
# ============================================================

print()
print("Loading PaddleOCR...")

try:

    ocr = PaddleOCR(
        lang="en",

        use_doc_orientation_classify=False,

        use_doc_unwarping=False,

        use_textline_orientation=False,

        device="cpu",

        enable_mkldnn=False
    )

    print("PaddleOCR loaded successfully.")
    print("PaddleOCR CPU mode: oneDNN disabled.")

except Exception as e:

    print()
    print("ERROR loading PaddleOCR:")
    print(e)
    print()

    raise SystemExit(1)


# ============================================================
# OCR NORMALIZATION
# ============================================================

def normalize_plate(text):

    if text is None:
        return ""

    text = str(text).upper()

    # Remove spaces and special characters
    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text


# ============================================================
# PLATE VALIDATION
# ============================================================

def is_valid_plate(text):

    if not text:
        return False

    # Typical Indian plate length.
    # Keep this flexible because OCR may miss characters.
    if len(text) < 4:
        return False

    if len(text) > 12:
        return False

    return True


# ============================================================
# PLATE PREPROCESSING
# ============================================================

def preprocess_plate(plate):

    if plate is None:
        return None

    if plate.size == 0:
        return None

    try:

        height, width = plate.shape[:2]

        if width <= 0 or height <= 0:
            return None

        # Upscale
        scale = 4

        resized = cv2.resize(
            plate,
            (
                width * scale,
                height * scale
            ),
            interpolation=cv2.INTER_CUBIC
        )

        # Convert to grayscale
        gray = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2GRAY
        )

        # Improve contrast
        gray = cv2.equalizeHist(gray)

        # Light denoise
        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            0
        )

        # Convert back to BGR.
        # PaddleOCR works reliably with 3-channel images.
        processed = cv2.cvtColor(
            gray,
            cv2.COLOR_GRAY2BGR
        )

        return processed

    except Exception as e:

        print("Preprocessing error:", e)

        return None


# ============================================================
# EXTRACT OCR RESULT
# ============================================================

def extract_ocr_result(result):

    best_text = ""
    best_score = 0.0

    try:

        # ----------------------------------------------------
        # PaddleOCR 3.x result object
        # ----------------------------------------------------

        data = None

        if hasattr(result, "json"):

            data = result.json

            if callable(data):
                data = data()

        # ----------------------------------------------------
        # Convert JSON string to dictionary
        # ----------------------------------------------------

        if isinstance(data, str):

            try:

                data = json.loads(data)

            except Exception:

                return "", 0.0

        # ----------------------------------------------------
        # Dictionary
        # ----------------------------------------------------

        if not isinstance(data, dict):

            return "", 0.0

        # PaddleOCR structure:
        #
        # {
        #   "res": {
        #       "rec_texts": [...],
        #       "rec_scores": [...]
        #   }
        # }

        res = data.get(
            "res",
            data
        )

        if not isinstance(res, dict):
            return "", 0.0

        texts = res.get(
            "rec_texts",
            []
        )

        scores = res.get(
            "rec_scores",
            []
        )

        if texts is None:
            texts = []

        if scores is None:
            scores = []

        # ----------------------------------------------------
        # Convert arrays to normal lists
        # ----------------------------------------------------

        try:
            texts = list(texts)
        except Exception:
            texts = []

        try:
            scores = list(scores)
        except Exception:
            scores = []

        # ----------------------------------------------------
        # Find best OCR result
        # ----------------------------------------------------

        for index, raw_text in enumerate(texts):

            if raw_text is None:
                continue

            text = normalize_plate(raw_text)

            if not text:
                continue

            score = 0.0

            if index < len(scores):

                try:
                    score = float(scores[index])

                except Exception:
                    score = 0.0

            # ------------------------------------------------
            # Keep strongest valid result
            # ------------------------------------------------

            if is_valid_plate(text):

                if score > best_score:

                    best_text = text
                    best_score = score

        return best_text, best_score

    except Exception as e:

        print(
            "OCR result parsing error:",
            str(e)
        )

        return "", 0.0


# ============================================================
# RUN PADDLEOCR
# ============================================================

def run_ocr(plate):

    if plate is None:
        return "", 0.0

    if plate.size == 0:
        return "", 0.0

    try:

        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        processed = preprocess_plate(
            plate
        )

        if processed is None:
            return "", 0.0

        # ----------------------------------------------------
        # PaddleOCR prediction
        # ----------------------------------------------------

        results = ocr.predict(
            processed
        )

        if results is None:
            return "", 0.0

        # ----------------------------------------------------
        # Read every OCR result
        # ----------------------------------------------------

        best_text = ""
        best_score = 0.0

        for result in results:

            text, score = extract_ocr_result(
                result
            )

            if text and score > best_score:

                best_text = text
                best_score = score

        return (
            best_text,
            best_score
        )

    except Exception as e:

        print()
        print(
            "OCR inference error:",
            str(e)
        )

        return "", 0.0


# ============================================================
# SAVE OCR HISTORY
# ============================================================

ocr_history = []


def save_ocr_history():

    try:

        with open(
            OCR_HISTORY_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                ocr_history,
                file,
                indent=4,
                ensure_ascii=False
            )

    except Exception as e:

        print(
            "Could not save OCR history:",
            e
        )


# ============================================================
# OPEN CAMERA
# ============================================================

print()
print("Opening webcam...")

camera = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

if not camera.isOpened():

    print()
    print("DirectShow camera open failed.")
    print("Trying default OpenCV camera backend...")
    print()

    camera.release()

    camera = cv2.VideoCapture(
        CAMERA_INDEX
    )


if not camera.isOpened():

    print()
    print("ERROR: Cannot open webcam.")
    print()
    print("Try:")
    print("CAMERA_INDEX = 1")
    print()

    raise SystemExit(1)


# ============================================================
# CAMERA SETTINGS
# ============================================================

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)

camera.set(
    cv2.CAP_PROP_FPS,
    30
)

# Reduce camera buffering
camera.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


actual_width = int(
    camera.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

actual_height = int(
    camera.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

camera_fps = camera.get(
    cv2.CAP_PROP_FPS
)


print()
print("Camera opened successfully.")
print("-" * 60)
print(
    f"Resolution : "
    f"{actual_width} x {actual_height}"
)
print(
    f"FPS        : "
    f"{camera_fps:.2f}"
)
print("-" * 60)


# ============================================================
# VARIABLES
# ============================================================

frame_count = 0

last_ocr_frame = -OCR_INTERVAL

last_plate_text = ""

last_ocr_confidence = 0.0

last_detector_confidence = 0.0

last_ocr_method = "PaddleOCR"

detected_plates = []

saved_frame_number = 0

saved_crop_number = 0

fps_start = time.time()

fps_counter = 0

display_fps = 0.0


# ============================================================
# START
# ============================================================

print()
print("REAL-TIME ANPR + OCR STARTED")
print("-" * 60)
print("Q / ESC = Exit")
print("S       = Save current frame")
print("-" * 60)
print()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = camera.read()

    if not success:

        print(
            "ERROR: Could not read camera frame."
        )

        break


    frame_count += 1
    fps_counter += 1


    # ========================================================
    # FPS
    # ========================================================

    elapsed = (
        time.time()
        - fps_start
    )

    if elapsed >= 1.0:

        display_fps = (
            fps_counter
            / elapsed
        )

        fps_counter = 0

        fps_start = time.time()


    # ========================================================
    # YOLO DETECTION
    # ========================================================

    try:

        results = detector.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )

    except Exception as e:

        print(
            "Detection error:",
            e
        )

        continue


    detected_plates = []


    # ========================================================
    # PROCESS DETECTIONS
    # ========================================================

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            try:

                detector_confidence = float(
                    box.conf[0]
                )

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

            except Exception:

                continue


            # ------------------------------------------------
            # Clamp coordinates
            # ------------------------------------------------

            x1 = max(
                0,
                x1
            )

            y1 = max(
                0,
                y1
            )

            x2 = min(
                frame.shape[1],
                x2
            )

            y2 = min(
                frame.shape[0],
                y2
            )


            width = x2 - x1
            height = y2 - y1


            if width < MIN_PLATE_WIDTH:
                continue

            if height < MIN_PLATE_HEIGHT:
                continue


            # ------------------------------------------------
            # Save latest detector confidence
            # ------------------------------------------------

            last_detector_confidence = (
                detector_confidence
            )


            # =================================================
            # PADDING
            # =================================================

            px1 = max(
                0,
                x1 - PLATE_PADDING
            )

            py1 = max(
                0,
                y1 - PLATE_PADDING
            )

            px2 = min(
                frame.shape[1],
                x2 + PLATE_PADDING
            )

            py2 = min(
                frame.shape[0],
                y2 + PLATE_PADDING
            )


            # =================================================
            # CROP PLATE
            # =================================================

            plate_crop = frame[
                py1:py2,
                px1:px2
            ]


            # =================================================
            # OCR
            # =================================================

            if (
                frame_count
                - last_ocr_frame
                >= OCR_INTERVAL
            ):

                text, ocr_confidence = run_ocr(
                    plate_crop
                )

                last_ocr_frame = frame_count


                if text:

                    # -----------------------------------------
                    # Confidence check
                    # -----------------------------------------

                    if (
                        ocr_confidence
                        >= MIN_OCR_CONFIDENCE
                    ):

                        last_plate_text = text

                        last_ocr_confidence = (
                            ocr_confidence
                        )


                        # -------------------------------------
                        # Save crop
                        # -------------------------------------

                        saved_crop_number += 1

                        crop_filename = os.path.join(
                            PLATE_CROP_DIR,
                            (
                                f"plate_"
                                f"{saved_crop_number:04d}_"
                                f"{text}.jpg"
                            )
                        )

                        # Prevent invalid Windows filename chars
                        crop_filename = re.sub(
                            r'[<>:"/\\|?*]',
                            "_",
                            crop_filename
                        )

                        cv2.imwrite(
                            crop_filename,
                            plate_crop
                        )


                        # -------------------------------------
                        # OCR history
                        # -------------------------------------

                        history_entry = {

                            "frame": frame_count,

                            "plate": text,

                            "ocr_confidence":
                                round(
                                    ocr_confidence * 100,
                                    2
                                ),

                            "detector_confidence":
                                round(
                                    detector_confidence * 100,
                                    2
                                ),

                            "method":
                                "PaddleOCR",

                            "timestamp":
                                time.strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                )
                        }


                        ocr_history.append(
                            history_entry
                        )


                        save_ocr_history()


                        # -------------------------------------
                        # Terminal output
                        # -------------------------------------

                        print(
                            f"[Frame {frame_count}] "
                            f"Plate: {text} | "
                            f"Detector: "
                            f"{detector_confidence * 100:.1f}% | "
                            f"OCR: "
                            f"{ocr_confidence * 100:.2f}%"
                        )


            # =================================================
            # DRAW DETECTION BOX
            # =================================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # =================================================
            # DETECTOR LABEL
            # =================================================

            detector_label = (
                "NUMBER PLATE "
                f"{detector_confidence * 100:.1f}%"
            )

            cv2.putText(
                frame,
                detector_label,
                (
                    x1,
                    max(
                        30,
                        y1 - 10
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )


            # =================================================
            # OCR LABEL
            # =================================================

            if last_plate_text:

                ocr_label = (
                    f"{last_plate_text} "
                    f"| OCR "
                    f"{last_ocr_confidence * 100:.1f}%"
                )

                cv2.putText(
                    frame,
                    ocr_label,
                    (
                        x1,
                        min(
                            frame.shape[0] - 10,
                            y2 + 30
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2
                )


            # =================================================
            # ADD DETECTION
            # =================================================

            detected_plates.append({

                "plate":
                    last_plate_text,

                "detector_confidence":
                    detector_confidence * 100,

                "ocr_confidence":
                    last_ocr_confidence * 100
            })


    # ========================================================
    # TOP INFORMATION PANEL
    # ========================================================

    cv2.putText(
        frame,
        "AEGISVISION - REAL-TIME ANPR",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Frame: {frame_count}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"FPS: {display_fps:.1f}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Plates detected: "
        f"{len(detected_plates)}",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DETECTOR STATUS
    # ========================================================

    cv2.putText(
        frame,
        "DETECTOR: ACTIVE",
        (20, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )


    # ========================================================
    # OCR STATUS
    # ========================================================

    if last_plate_text:

        cv2.putText(
            frame,
            "OCR: ACTIVE",
            (20, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "OCR: WAITING",
            (20, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )


    # ========================================================
    # CURRENT RESULT PANEL
    # ========================================================

    if last_plate_text:

        cv2.rectangle(
            frame,
            (15, 215),
            (370, 310),
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            "DETECTED PLATE",
            (30, 245),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            last_plate_text,
            (30, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.90,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            (
                f"OCR: "
                f"{last_ocr_confidence * 100:.1f}%"
            ),
            (30, 302),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1
        )


    # ========================================================
    # CONTROLS
    # ========================================================

    cv2.putText(
        frame,
        "Q / ESC = Exit",
        (
            20,
            frame.shape[0] - 40
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "S = Save Frame",
        (
            250,
            frame.shape[0] - 40
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------

    if (
        key == ord("q")
        or key == ord("Q")
        or key == 27
    ):

        break


    # --------------------------------------------------------
    # SAVE FRAME
    # --------------------------------------------------------

    if (
        key == ord("s")
        or key == ord("S")
    ):

        saved_frame_number += 1

        filename = os.path.join(
            OUTPUT_DIR,
            f"frame_{saved_frame_number:04d}.jpg"
        )

        cv2.imwrite(
            filename,
            frame
        )

        print()
        print("Frame saved:")
        print(filename)
        print()


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("PHASE 6.2 COMPLETED")
print("=" * 60)

print()

print(
    "Total frames processed :",
    frame_count
)

print(
    "Last detected plate    :",
    last_plate_text or "None"
)

print(
    "Last OCR confidence    : "
    f"{last_ocr_confidence * 100:.2f}%"
)

print(
    "Last OCR method        :",
    last_ocr_method if last_plate_text else "None"
)

print(
    "OCR readings collected :",
    len(ocr_history)
)

print()

print("Saved frames directory:")
print(OUTPUT_DIR)

print()

print("Saved plate crops:")
print(PLATE_CROP_DIR)

print()

print("OCR history:")
print(OCR_HISTORY_FILE)

print()

print("=" * 60)
print("REAL-TIME OCR INTEGRATION FINISHED.")
print("=" * 60)