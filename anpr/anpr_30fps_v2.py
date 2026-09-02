"""
======================================================================
AEGISVISION - ANPR
PHASE 6.5 V2 - HIGH SPEED REAL-TIME ANPR
TARGET: 30+ PROCESSING FPS

Optimizations:
    - YOLO detection every fixed number of frames
    - No repeated YOLO inference when tracking is lost
    - Smaller YOLO inference resolution
    - Lightweight bounding-box prediction between detections
    - Asynchronous OCR worker
    - PaddleOCR primary
    - Tesseract fallback
    - Multi-frame OCR history
    - Stable plate voting
    - 640x480 camera
======================================================================
"""

import os
import cv2
import json
import time
import re
import threading
import queue
from collections import Counter
from difflib import SequenceMatcher

from ultralytics import YOLO
import pytesseract


# ======================================================================
# CONFIGURATION
# ======================================================================

MODEL_PATH = "models/plate_detector.pt"

OUTPUT_DIR = "output/phase6_5_v2"

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Camera
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 60

# YOLO optimization
YOLO_IMG_SIZE = 256

# Run YOLO once every N frames
# Higher = faster FPS
DETECTION_INTERVAL = 10

# Confidence threshold
YOLO_CONFIDENCE = 0.25

# Maximum number of consecutive frames where a prediction
# can continue without a fresh YOLO detection
MAX_TRACK_FRAMES = 30

# OCR
OCR_INTERVAL = 60

# Maximum OCR queue size
OCR_QUEUE_SIZE = 2

# OCR history
MAX_OCR_HISTORY = 20

# Stability
MIN_PLATE_LENGTH = 4
MAX_PLATE_LENGTH = 12

# Display
WINDOW_NAME = "AEGISVISION ANPR - Phase 6.5 V2"


# ======================================================================
# GLOBAL OCR VARIABLES
# ======================================================================

ocr_queue = queue.Queue(maxsize=OCR_QUEUE_SIZE)

ocr_thread_running = True

latest_ocr_text = ""
latest_ocr_confidence = 0.0
latest_ocr_engine = ""

ocr_lock = threading.Lock()

ocr_history = []


# ======================================================================
# GLOBAL OCR OBJECT
# ======================================================================

paddle_ocr = None


# ======================================================================
# CREATE OUTPUT DIRECTORY
# ======================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ======================================================================
# TESSERACT SETUP
# ======================================================================

def setup_tesseract():

    print()
    print("Checking Tesseract OCR...")

    if os.path.exists(TESSERACT_PATH):

        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

        print("Tesseract executable found:")
        print(TESSERACT_PATH)

        try:

            version = pytesseract.get_tesseract_version()

            print("Tesseract loaded successfully.")
            print("Version:", version)

            return True

        except Exception as e:

            print("Tesseract found but failed to load:")
            print(e)

            return False

    print("Tesseract executable not found.")
    return False


# ======================================================================
# PLATE TEXT CLEANING
# ======================================================================

def clean_plate_text(text):

    if text is None:
        return ""

    text = str(text).upper()

    # Remove everything except letters and numbers
    text = re.sub(r"[^A-Z0-9]", "", text)

    # Common OCR corrections
    replacements = {
        "O": "0",
        "I": "1",
        "L": "1",
    }

    # Do not blindly replace every character.
    # Only use replacements for very short/noisy OCR results.
    if len(text) <= 5:

        for old, new in replacements.items():
            text = text.replace(old, new)

    return text


# ======================================================================
# VALIDATE PLATE
# ======================================================================

def is_valid_plate(text):

    if not text:
        return False

    length = len(text)

    if length < MIN_PLATE_LENGTH:
        return False

    if length > MAX_PLATE_LENGTH:
        return False

    # Must contain both letters and numbers
    has_letter = any(c.isalpha() for c in text)
    has_number = any(c.isdigit() for c in text)

    if not has_letter or not has_number:
        return False

    return True


# ======================================================================
# PREPROCESS PLATE
# ======================================================================

def preprocess_plate(crop):

    if crop is None or crop.size == 0:
        return None

    # Resize
    height, width = crop.shape[:2]

    if width < 200:

        scale = 2.0

        crop = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

    else:

        crop = cv2.resize(
            crop,
            None,
            fx=1.5,
            fy=1.5,
            interpolation=cv2.INTER_CUBIC
        )

    # Convert grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Mild denoise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Improve contrast
    gray = cv2.equalizeHist(gray)

    return gray


# ======================================================================
# TESSERACT OCR
# ======================================================================

def run_tesseract(image):

    if image is None:
        return "", 0.0

    try:

        config = (
            "--psm 7 "
            "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

        data = pytesseract.image_to_data(
            image,
            config=config,
            output_type=pytesseract.Output.DICT
        )

        texts = []
        confidences = []

        for i in range(len(data["text"])):

            txt = data["text"][i].strip()

            if not txt:
                continue

            try:
                conf = float(data["conf"][i])
            except:
                conf = 0.0

            if conf >= 0:

                texts.append(txt)
                confidences.append(conf)

        if not texts:
            return "", 0.0

        text = "".join(texts)

        confidence = sum(confidences) / len(confidences)

        text = clean_plate_text(text)

        if not is_valid_plate(text):
            return "", confidence

        return text, confidence

    except Exception as e:

        print("[TESSERACT ERROR]", e)

        return "", 0.0


# ======================================================================
# PADDLE OCR
# ======================================================================

def run_paddleocr(image):

    global paddle_ocr

    if image is None:
        return "", 0.0

    if paddle_ocr is None:
        return "", 0.0

    try:

        results = paddle_ocr.predict(image)

        best_text = ""
        best_conf = 0.0

        for result in results:

            data = result.json

            if callable(data):
                data = data()

            if not isinstance(data, dict):
                continue

            res = data.get("res", data)

            texts = res.get("rec_texts", [])
            scores = res.get("rec_scores", [])

            for text, score in zip(texts, scores):

                text = clean_plate_text(text)

                try:
                    score = float(score) * 100.0
                except:
                    score = 0.0

                if not is_valid_plate(text):
                    continue

                if score > best_conf:

                    best_text = text
                    best_conf = score

        return best_text, best_conf

    except Exception as e:

        print("[PADDLE OCR ERROR]", e)

        return "", 0.0


# ======================================================================
# OCR WORKER
# ======================================================================

def ocr_worker():

    global paddle_ocr
    global latest_ocr_text
    global latest_ocr_confidence
    global latest_ocr_engine
    global ocr_thread_running
    global ocr_history

    print()
    print("[OCR WORKER] Loading PaddleOCR...")

    try:

        from paddleocr import PaddleOCR

        paddle_ocr = PaddleOCR(
            lang="en",
            device="cpu",
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )

        print("[OCR WORKER] PaddleOCR loaded.")
        print("[OCR WORKER] CPU mode enabled.")
        print("[OCR WORKER] MKL-DNN disabled.")

    except Exception as e:

        print("[OCR WORKER] PaddleOCR failed to load:")
        print(e)

        paddle_ocr = None

    while ocr_thread_running:

        try:

            item = ocr_queue.get(timeout=0.2)

        except queue.Empty:

            continue

        if item is None:
            break

        frame_number, plate_crop = item

        processed = preprocess_plate(plate_crop)

        if processed is None:
            continue

        # --------------------------------------------------------------
        # PaddleOCR first
        # --------------------------------------------------------------

        text = ""
        confidence = 0.0
        engine = ""

        if paddle_ocr is not None:

            text, confidence = run_paddleocr(processed)

            if text:

                engine = "PaddleOCR"

        # --------------------------------------------------------------
        # Tesseract fallback
        # --------------------------------------------------------------

        if not text:

            text, confidence = run_tesseract(processed)

            if text:

                engine = "Tesseract"

        # --------------------------------------------------------------
        # Store result
        # --------------------------------------------------------------

        if text:

            with ocr_lock:

                latest_ocr_text = text
                latest_ocr_confidence = confidence
                latest_ocr_engine = engine

                ocr_history.append(
                    {
                        "frame": frame_number,
                        "plate": text,
                        "confidence": round(confidence, 2),
                        "engine": engine,
                        "timestamp": time.time()
                    }
                )

                if len(ocr_history) > MAX_OCR_HISTORY:

                    ocr_history.pop(0)

        ocr_queue.task_done()

    print("[OCR WORKER] Stopped.")


# ======================================================================
# IOU
# ======================================================================

def calculate_iou(box1, box2):

    if box1 is None or box2 is None:
        return 0.0

    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2

    inter_x1 = max(x1, a1)
    inter_y1 = max(y1, b1)

    inter_x2 = min(x2, a2)
    inter_y2 = min(y2, b2)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)

    intersection = inter_width * inter_height

    area1 = max(0, x2 - x1) * max(0, y2 - y1)
    area2 = max(0, a2 - a1) * max(0, b2 - b1)

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ======================================================================
# PREDICT BOX BETWEEN YOLO DETECTIONS
# ======================================================================

def predict_box(
    current_box,
    previous_box,
    frames_since_detection,
    frame_width,
    frame_height
):

    if current_box is None:
        return None

    if previous_box is None:
        return current_box

    if frames_since_detection <= 0:
        return current_box

    x1, y1, x2, y2 = current_box

    px1, py1, px2, py2 = previous_box

    # Calculate previous center
    current_cx = (x1 + x2) / 2
    current_cy = (y1 + y2) / 2

    previous_cx = (px1 + px2) / 2
    previous_cy = (py1 + py2) / 2

    # Estimate velocity
    vx = current_cx - previous_cx
    vy = current_cy - previous_cy

    # Limit velocity to avoid wild jumps
    vx = max(-15, min(15, vx))
    vy = max(-15, min(15, vy))

    shift_x = vx * frames_since_detection
    shift_y = vy * frames_since_detection

    new_x1 = int(x1 + shift_x)
    new_y1 = int(y1 + shift_y)
    new_x2 = int(x2 + shift_x)
    new_y2 = int(y2 + shift_y)

    # Clamp to frame
    new_x1 = max(0, min(frame_width - 1, new_x1))
    new_y1 = max(0, min(frame_height - 1, new_y1))

    new_x2 = max(0, min(frame_width - 1, new_x2))
    new_y2 = max(0, min(frame_height - 1, new_y2))

    return (
        new_x1,
        new_y1,
        new_x2,
        new_y2
    )


# ======================================================================
# DETECT PLATE
# ======================================================================

def detect_plate(model, frame):

    try:

        results = model.predict(
            frame,
            imgsz=YOLO_IMG_SIZE,
            conf=YOLO_CONFIDENCE,
            verbose=False,
            device="cpu"
        )

        best_box = None
        best_confidence = 0.0

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                confidence = float(box.conf[0])

                if confidence < best_confidence:
                    continue

                coordinates = box.xyxy[0].cpu().numpy()

                x1, y1, x2, y2 = coordinates

                best_box = (
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                )

                best_confidence = confidence

        return best_box, best_confidence

    except Exception as e:

        print("[YOLO ERROR]", e)

        return None, 0.0


# ======================================================================
# CROP PLATE
# ======================================================================

def crop_plate(frame, box):

    if box is None:
        return None

    height, width = frame.shape[:2]

    x1, y1, x2, y2 = box

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return crop


# ======================================================================
# OCR STABILITY
# ======================================================================

def calculate_similarity(a, b):

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio() * 100.0


def get_stable_plate():

    if not ocr_history:
        return "", 0.0, 0, 0.0

    valid_items = []

    for item in ocr_history:

        text = item["plate"]

        if is_valid_plate(text):

            valid_items.append(item)

    if not valid_items:
        return "", 0.0, 0, 0.0

    texts = [item["plate"] for item in valid_items]

    counts = Counter(texts)

    stable_text, votes = counts.most_common(1)[0]

    confidence_values = [
        item["confidence"]
        for item in valid_items
        if item["plate"] == stable_text
    ]

    if confidence_values:
        confidence = sum(confidence_values) / len(confidence_values)
    else:
        confidence = 0.0

    similarities = []

    for text in texts:

        if text != stable_text:

            similarities.append(
                calculate_similarity(
                    stable_text,
                    text
                )
            )

    if similarities:

        fuzzy_similarity = sum(similarities) / len(similarities)

    else:

        fuzzy_similarity = 100.0

    return (
        stable_text,
        confidence,
        votes,
        fuzzy_similarity
    )


# ======================================================================
# SAVE RESULTS
# ======================================================================

def save_json(filename, data):

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    try:

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                indent=4
            )

        return path

    except Exception as e:

        print("[SAVE ERROR]", e)

        return ""


# ======================================================================
# MAIN
# ======================================================================

def main():

    global latest_ocr_text
    global latest_ocr_confidence
    global latest_ocr_engine
    global ocr_thread_running

    print()
    print("=" * 70)
    print("AEGISVISION - ANPR")
    print("PHASE 6.5 V2 - HIGH SPEED REAL-TIME ANPR")
    print("TARGET: 30+ PROCESSING FPS")
    print()
    print("YOLO + ASYNC PADDLEOCR")
    print("FIXED DETECTION INTERVAL")
    print("LIGHTWEIGHT BOX PREDICTION")
    print("=" * 70)

    # --------------------------------------------------------------
    # Load YOLO
    # --------------------------------------------------------------

    print()
    print("Loading YOLO plate detector...")

    try:

        model = YOLO(MODEL_PATH)

        print("YOLO detector loaded successfully.")

    except Exception as e:

        print()
        print("ERROR: Could not load YOLO model.")
        print(e)

        return

    # --------------------------------------------------------------
    # Tesseract
    # --------------------------------------------------------------

    setup_tesseract()

    # --------------------------------------------------------------
    # Start OCR thread
    # --------------------------------------------------------------

    print()
    print("Starting asynchronous OCR worker...")

    worker = threading.Thread(
        target=ocr_worker,
        daemon=True
    )

    worker.start()

    # --------------------------------------------------------------
    # Open camera
    # --------------------------------------------------------------

    print()
    print("Opening webcam...")

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():

        print()
        print("ERROR: Could not open webcam.")

        ocr_thread_running = False

        return

    # --------------------------------------------------------------
    # Camera settings
    # --------------------------------------------------------------

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        CAMERA_FPS
    )

    # MJPG
    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(
            *"MJPG"
        )
    )

    actual_width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    actual_height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    actual_camera_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    print()
    print("Camera opened successfully.")
    print("-" * 70)
    print(
        f"Resolution : "
        f"{actual_width} x {actual_height}"
    )

    print(
        f"Requested FPS : "
        f"{CAMERA_FPS}"
    )

    print(
        f"Camera FPS    : "
        f"{actual_camera_fps:.2f}"
    )

    print("-" * 70)

    # --------------------------------------------------------------
    # Runtime variables
    # --------------------------------------------------------------

    total_frames = 0

    detection_runs = 0

    total_detections = 0

    ocr_attempts = 0

    ocr_successes = 0

    ocr_errors = 0

    current_box = None

    previous_box = None

    last_detection_box = None

    last_detection_confidence = 0.0

    frames_since_detection = 0

    # FPS
    fps_start_time = time.perf_counter()

    fps_frame_count = 0

    processing_fps = 0.0

    # Current FPS shown on screen
    display_fps = 0.0

    # --------------------------------------------------------------
    # Start
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("PHASE 6.5 V2 HIGH-SPEED ANPR STARTED")
    print("=" * 70)
    print()
    print("Controls:")
    print("  Q / ESC = Quit")
    print("  S       = Save current plate crop")
    print("  R       = Reset tracking")
    print()
    print(
        f"YOLO interval : "
        f"Every {DETECTION_INTERVAL} frames"
    )

    print(
        f"YOLO image size : "
        f"{YOLO_IMG_SIZE}"
    )

    print(
        f"OCR interval : "
        f"Every {OCR_INTERVAL} frames"
    )

    print()

    # --------------------------------------------------------------
    # Main loop
    # --------------------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:

            print("Camera frame read failed.")

            break

        total_frames += 1

        fps_frame_count += 1

        frame_height, frame_width = frame.shape[:2]

        # ==========================================================
        # FIXED YOLO SCHEDULING
        # ==========================================================

        run_detection = (
            total_frames == 1
            or total_frames % DETECTION_INTERVAL == 0
        )

        # ----------------------------------------------------------
        # YOLO detection
        # ----------------------------------------------------------

        if run_detection:

            detection_runs += 1

            detected_box, detected_confidence = detect_plate(
                model,
                frame
            )

            if detected_box is not None:

                total_detections += 1

                previous_box = current_box

                current_box = detected_box

                last_detection_box = detected_box

                last_detection_confidence = (
                    detected_confidence
                )

                frames_since_detection = 0

            else:

                frames_since_detection += (
                    DETECTION_INTERVAL
                )

        else:

            frames_since_detection += 1

        # ==========================================================
        # LIGHTWEIGHT TRACKING / PREDICTION
        # ==========================================================

        display_box = current_box

        if (
            current_box is not None
            and previous_box is not None
            and frames_since_detection > 0
            and frames_since_detection <= MAX_TRACK_FRAMES
        ):

            display_box = predict_box(
                current_box,
                previous_box,
                frames_since_detection,
                frame_width,
                frame_height
            )

        # ----------------------------------------------------------
        # If tracking has been lost for too long
        # ----------------------------------------------------------

        if frames_since_detection > MAX_TRACK_FRAMES:

            current_box = None
            previous_box = None
            display_box = None

        # ==========================================================
        # OCR
        # ==========================================================

        if (
            display_box is not None
            and total_frames % OCR_INTERVAL == 0
        ):

            plate_crop = crop_plate(
                frame,
                display_box
            )

            if plate_crop is not None:

                ocr_attempts += 1

                try:

                    ocr_queue.put_nowait(
                        (
                            total_frames,
                            plate_crop.copy()
                        )
                    )

                except queue.Full:

                    # Skip OCR rather than blocking the main loop.
                    pass

        # ==========================================================
        # READ LATEST OCR RESULT
        # ==========================================================

        with ocr_lock:

            current_ocr_text = latest_ocr_text

            current_ocr_confidence = (
                latest_ocr_confidence
            )

            current_ocr_engine = (
                latest_ocr_engine
            )

            history_length = len(
                ocr_history
            )

        # ==========================================================
        # STABLE RESULT
        # ==========================================================

        (
            stable_plate,
            stable_confidence,
            stable_votes,
            fuzzy_similarity
        ) = get_stable_plate()

        # ==========================================================
        # FPS CALCULATION
        # ==========================================================

        elapsed = (
            time.perf_counter()
            - fps_start_time
        )

        if elapsed >= 1.0:

            processing_fps = (
                fps_frame_count / elapsed
            )

            display_fps = processing_fps

            fps_frame_count = 0

            fps_start_time = time.perf_counter()

        # ==========================================================
        # DRAW DETECTION BOX
        # ==========================================================

        if display_box is not None:

            x1, y1, x2, y2 = display_box

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = (
                f"PLATE "
                f"{last_detection_confidence * 100:.0f}%"
            )

            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

        # ==========================================================
        # DRAW OCR RESULT
        # ==========================================================

        if current_ocr_text:

            cv2.putText(
                frame,
                f"OCR: {current_ocr_text}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                (
                    f"{current_ocr_engine} "
                    f"{current_ocr_confidence:.1f}%"
                ),
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

        # ==========================================================
        # DRAW STABLE PLATE
        # ==========================================================

        if stable_plate:

            cv2.putText(
                frame,
                f"STABLE: {stable_plate}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                (
                    f"Confidence: "
                    f"{stable_confidence:.1f}% "
                    f"Votes: {stable_votes}"
                ),
                (10, 118),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

        # ==========================================================
        # DRAW FPS
        # ==========================================================

        cv2.putText(
            frame,
            f"Processing FPS: {display_fps:.1f}",
            (10, frame_height - 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            (
                f"YOLO: "
                f"{detection_runs} | "
                f"OCR: {ocr_attempts}"
            ),
            (10, frame_height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

        # ==========================================================
        # DISPLAY
        # ==========================================================

        cv2.imshow(
            WINDOW_NAME,
            frame
        )

        # ==========================================================
        # KEYBOARD
        # ==========================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:

            break

        # ----------------------------------------------------------
        # Save crop
        # ----------------------------------------------------------

        if key == ord("s"):

            if display_box is not None:

                crop = crop_plate(
                    frame,
                    display_box
                )

                if crop is not None:

                    filename = os.path.join(
                        OUTPUT_DIR,
                        f"plate_{total_frames}.jpg"
                    )

                    cv2.imwrite(
                        filename,
                        crop
                    )

                    print(
                        f"[SAVE] Plate crop saved: "
                        f"{filename}"
                    )

        # ----------------------------------------------------------
        # Reset tracking
        # ----------------------------------------------------------

        if key == ord("r"):

            current_box = None

            previous_box = None

            last_detection_box = None

            frames_since_detection = 0

            print("[TRACKING] Reset.")

    # ==================================================================
    # STOP
    # ==================================================================

    print()
    print("Stopping camera...")

    cap.release()

    cv2.destroyAllWindows()

    # Stop OCR thread
    ocr_thread_running = False

    try:

        ocr_queue.put_nowait(None)

    except queue.Full:

        pass

    worker.join(timeout=3)

    # ==================================================================
    # FINAL RESULTS
    # ==================================================================

    stable_plate, stable_confidence, stable_votes, fuzzy_similarity = (
        get_stable_plate()
    )

    # --------------------------------------------------------------
    # Count OCR successes
    # --------------------------------------------------------------

    with ocr_lock:

        ocr_successes = len(ocr_history)

    # --------------------------------------------------------------
    # Final ANPR result
    # --------------------------------------------------------------

    final_result = {
        "phase": "6.5_v2",
        "target_processing_fps": 30,
        "total_frames_processed": total_frames,
        "yolo_detection_runs": detection_runs,
        "total_yolo_detections": total_detections,
        "ocr_attempts": ocr_attempts,
        "ocr_successes": ocr_successes,
        "ocr_errors": ocr_errors,
        "camera_fps": round(
            actual_camera_fps,
            2
        ),
        "average_processing_fps": round(
            processing_fps,
            2
        ),
        "last_ocr_plate": latest_ocr_text,
        "last_ocr_confidence": round(
            latest_ocr_confidence,
            2
        ),
        "last_ocr_engine": latest_ocr_engine,
        "stable_plate": stable_plate,
        "stable_confidence": round(
            stable_confidence,
            2
        ),
        "stable_votes": stable_votes,
        "fuzzy_similarity": round(
            fuzzy_similarity,
            2
        ),
        "ocr_readings_collected": len(
            ocr_history
        ),
        "configuration": {
            "camera_resolution": (
                f"{actual_width}x{actual_height}"
            ),
            "requested_camera_fps": CAMERA_FPS,
            "yolo_image_size": YOLO_IMG_SIZE,
            "detection_interval": DETECTION_INTERVAL,
            "ocr_interval": OCR_INTERVAL,
            "async_ocr": True,
            "paddleocr_enabled": paddle_ocr is not None,
            "tesseract_path": TESSERACT_PATH
        }
    }

    # ==================================================================
    # SAVE OCR HISTORY
    # ==================================================================

    ocr_history_path = save_json(
        "ocr_history.json",
        ocr_history
    )

    # ==================================================================
    # SAVE STABILITY
    # ==================================================================

    stability_data = {
        "stable_plate": stable_plate,
        "stable_confidence": round(
            stable_confidence,
            2
        ),
        "stable_votes": stable_votes,
        "fuzzy_similarity": round(
            fuzzy_similarity,
            2
        ),
        "readings": ocr_history
    }

    stability_path = save_json(
        "ocr_stability_clusters.json",
        stability_data
    )

    # ==================================================================
    # SAVE FINAL RESULT
    # ==================================================================

    final_path = save_json(
        "final_anpr_result.json",
        final_result
    )

    # ==================================================================
    # TERMINAL REPORT
    # ==================================================================

    print()
    print("=" * 70)
    print("PHASE 6.5 V2 TEST RESULT")
    print("=" * 70)

    print(
        f"Total frames processed : "
        f"{total_frames}"
    )

    print(
        f"YOLO detection runs    : "
        f"{detection_runs}"
    )

    print(
        f"Total YOLO detections  : "
        f"{total_detections}"
    )

    print(
        f"OCR attempts           : "
        f"{ocr_attempts}"
    )

    print(
        f"OCR successes          : "
        f"{ocr_successes}"
    )

    print(
        f"OCR errors             : "
        f"{ocr_errors}"
    )

    print(
        f"Camera FPS             : "
        f"{actual_camera_fps:.2f}"
    )

    print(
        f"Average processing FPS : "
        f"{processing_fps:.2f}"
    )

    print()

    print(
        f"Last OCR plate         : "
        f"{latest_ocr_text}"
    )

    print(
        f"Last OCR confidence    : "
        f"{latest_ocr_confidence:.2f}%"
    )

    print(
        f"Last OCR engine        : "
        f"{latest_ocr_engine}"
    )

    print()

    print(
        f"Stable plate           : "
        f"{stable_plate}"
    )

    print(
        f"Stable confidence      : "
        f"{stable_confidence:.2f}%"
    )

    print(
        f"Stable votes           : "
        f"{stable_votes}"
    )

    print(
        f"Fuzzy similarity       : "
        f"{fuzzy_similarity:.2f}%"
    )

    print(
        f"OCR readings collected : "
        f"{len(ocr_history)}"
    )

    print()

    print("Output directory:")
    print(
        os.path.abspath(
            OUTPUT_DIR
        )
    )

    print()

    print("OCR history:")
    print(
        os.path.abspath(
            ocr_history_path
        )
    )

    print()

    print("Stability clusters:")
    print(
        os.path.abspath(
            stability_path
        )
    )

    print()

    print("Final ANPR result:")
    print(
        os.path.abspath(
            final_path
        )
    )

    print("=" * 70)

    # ==================================================================
    # PASS / FAIL
    # ==================================================================

    if processing_fps >= 30:

        print()
        print("✓ 30+ FPS TARGET ACHIEVED!")
        print(
            f"Processing FPS = "
            f"{processing_fps:.2f}"
        )

        print()
        print("PHASE 6.5 V2 HIGH-SPEED TARGET PASSED.")

    else:

        print()
        print("⚠ 30 FPS TARGET NOT YET ACHIEVED.")

        print(
            f"Current processing FPS = "
            f"{processing_fps:.2f}"
        )

        print(
            f"Required processing FPS = "
            f"30.00"
        )

        print()
        print(
            "PHASE 6.5 V2 NEEDS FURTHER OPTIMIZATION."
        )

    print("=" * 70)


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":

    main()