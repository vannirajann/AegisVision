import os
import re
import cv2
import json
import time
import queue
import threading
import numpy as np
import pytesseract

from collections import Counter, deque
from difflib import SequenceMatcher

from ultralytics import YOLO
from paddleocr import PaddleOCR


# ============================================================
# AEGISVISION - ANPR
# PHASE 6.5 - HIGH SPEED REAL-TIME ANPR
# TARGET: 30+ PROCESSING FPS
# ============================================================

MODEL_PATH = "models/plate_detector.pt"

OUTPUT_DIR = "output/phase6_5"
CROP_DIR = os.path.join(OUTPUT_DIR, "plate_crops")

CAMERA_INDEX = 0

# Camera
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
TARGET_CAMERA_FPS = 60

# YOLO processing resolution
DETECTION_WIDTH = 320
DETECTION_HEIGHT = 240

# YOLO
YOLO_CONFIDENCE = 0.25
DETECTION_INTERVAL = 5

# Tracking
IOU_THRESHOLD = 0.20
MAX_MISSES = 12

# OCR
OCR_INTERVAL = 30
MIN_PLATE_WIDTH = 30
MIN_PLATE_HEIGHT = 12
MIN_OCR_CONFIDENCE = 40.0

# OCR history
OCR_HISTORY_SIZE = 20
MIN_STABLE_VOTES = 3
FUZZY_THRESHOLD = 0.70

# OCR queue
OCR_QUEUE_SIZE = 2


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)


# ============================================================
# GLOBAL OCR STATE
# ============================================================

ocr_queue = queue.Queue(maxsize=OCR_QUEUE_SIZE)
ocr_results = queue.Queue()

ocr_stop_event = threading.Event()

latest_ocr_text = ""
latest_ocr_confidence = 0.0
latest_ocr_engine = ""

ocr_history = deque(maxlen=OCR_HISTORY_SIZE)

ocr_attempts = 0
ocr_successes = 0
ocr_errors = 0

ocr_lock = threading.Lock()


# ============================================================
# TESSERACT PATH
# ============================================================

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ============================================================
# CLEAN PLATE TEXT
# ============================================================

def clean_plate_text(text):

    if not text:
        return ""

    text = str(text).upper()

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text


# ============================================================
# IOU
# ============================================================

def calculate_iou(box1, box2):

    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2

    ix1 = max(x1, a1)
    iy1 = max(y1, b1)

    ix2 = min(x2, a2)
    iy2 = min(y2, b2)

    iw = max(
        0,
        ix2 - ix1
    )

    ih = max(
        0,
        iy2 - iy1
    )

    intersection = iw * ih

    area1 = max(
        0,
        x2 - x1
    ) * max(
        0,
        y2 - y1
    )

    area2 = max(
        0,
        a2 - a1
    ) * max(
        0,
        b2 - b1
    )

    union = (
        area1 +
        area2 -
        intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# FUZZY SIMILARITY
# ============================================================

def fuzzy_similarity(a, b):

    if not a or not b:
        return 0.0

    return SequenceMatcher(
        None,
        a,
        b
    ).ratio()


# ============================================================
# OCR PREPROCESSING
# ============================================================

def preprocess_plate(plate):

    if plate is None:
        return None

    if plate.size == 0:
        return None

    gray = cv2.cvtColor(
        plate,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        None,
        fx=2.0,
        fy=2.0,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    processed = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return processed


# ============================================================
# EXTRACT PADDLEOCR RESULT
# ============================================================

def extract_paddle_result(result):

    best_text = ""
    best_conf = 0.0

    try:

        if result is None:
            return "", 0.0

        # ----------------------------------------------------
        # New PaddleOCR result object
        # ----------------------------------------------------

        if hasattr(result, "json"):

            data = result.json

            if callable(data):
                data = data()

            if isinstance(data, str):

                data = json.loads(data)

            if isinstance(data, list):

                data = (
                    data[0]
                    if data
                    else {}
                )

            if isinstance(data, dict):

                rec_texts = data.get(
                    "rec_texts",
                    []
                )

                rec_scores = data.get(
                    "rec_scores",
                    []
                )

                for i, text in enumerate(
                    rec_texts
                ):

                    try:

                        score = float(
                            rec_scores[i]
                        )

                    except Exception:

                        score = 0.0

                    text = clean_plate_text(
                        text
                    )

                    if (
                        text
                        and score > best_conf
                    ):

                        best_text = text

                        best_conf = (
                            score * 100.0
                        )

        # ----------------------------------------------------
        # Dictionary format
        # ----------------------------------------------------

        if isinstance(
            result,
            dict
        ):

            rec_texts = result.get(
                "rec_texts",
                []
            )

            rec_scores = result.get(
                "rec_scores",
                []
            )

            for i, text in enumerate(
                rec_texts
            ):

                try:

                    score = float(
                        rec_scores[i]
                    )

                except Exception:

                    score = 0.0

                text = clean_plate_text(
                    text
                )

                if (
                    text
                    and score > best_conf
                ):

                    best_text = text

                    best_conf = (
                        score * 100.0
                    )

    except Exception:
        pass

    return (
        best_text,
        best_conf
    )


# ============================================================
# OCR WORKER
# ============================================================

def ocr_worker():

    global latest_ocr_text
    global latest_ocr_confidence
    global latest_ocr_engine

    global ocr_attempts
    global ocr_successes
    global ocr_errors

    print(
        "\n[OCR WORKER] Loading PaddleOCR..."
    )

    paddle_ocr = None

    try:

        paddle_ocr = PaddleOCR(
            lang="en",
            device="cpu",
            enable_mkldnn=False
        )

        print(
            "[OCR WORKER] PaddleOCR loaded."
        )

        print(
            "[OCR WORKER] CPU mode enabled."
        )

        print(
            "[OCR WORKER] MKL-DNN disabled."
        )

    except Exception as e:

        print(
            "[OCR WORKER] PaddleOCR loading failed:"
        )

        print(e)

    while not ocr_stop_event.is_set():

        try:

            item = ocr_queue.get(
                timeout=0.1
            )

        except queue.Empty:

            continue

        if item is None:

            try:
                ocr_queue.task_done()
            except Exception:
                pass

            break

        crop, frame_number = item

        try:

            ocr_attempts += 1

            if (
                crop is None
                or crop.size == 0
            ):

                ocr_queue.task_done()

                continue

            processed = preprocess_plate(
                crop
            )

            if processed is None:

                ocr_queue.task_done()

                continue

            text = ""
            confidence = 0.0
            engine = ""

            # =================================================
            # PADDLEOCR
            # =================================================

            if paddle_ocr is not None:

                try:

                    result = paddle_ocr.predict(
                        processed
                    )

                    if result:

                        for res in result:

                            t, c = (
                                extract_paddle_result(
                                    res
                                )
                            )

                            if (
                                t
                                and c > confidence
                            ):

                                text = t

                                confidence = c

                                engine = (
                                    "PaddleOCR"
                                )

                except Exception:
                    pass

            # =================================================
            # TESSERACT FALLBACK
            # =================================================

            if (
                not text
                or confidence <
                MIN_OCR_CONFIDENCE
            ):

                try:

                    config = (
                        "--oem 3 "
                        "--psm 7 "
                        "-c "
                        "tessedit_char_whitelist="
                        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                    )

                    tess_text = (
                        pytesseract.image_to_string(
                            processed,
                            config=config
                        )
                    )

                    tess_text = (
                        clean_plate_text(
                            tess_text
                        )
                    )

                    if tess_text:

                        text = tess_text

                        confidence = max(
                            confidence,
                            50.0
                        )

                        engine = (
                            "Tesseract"
                        )

                except Exception:
                    pass

            # =================================================
            # STORE OCR RESULT
            # =================================================

            if text:

                ocr_successes += 1

                record = {

                    "frame": frame_number,

                    "plate": text,

                    "confidence": round(
                        confidence,
                        2
                    ),

                    "engine": engine,

                    "timestamp": time.time()
                }

                with ocr_lock:

                    latest_ocr_text = text

                    latest_ocr_confidence = (
                        confidence
                    )

                    latest_ocr_engine = (
                        engine
                    )

                    ocr_history.append(
                        record
                    )

                try:

                    ocr_results.put_nowait(
                        record
                    )

                except queue.Full:

                    pass

            ocr_queue.task_done()

        except Exception as e:

            ocr_errors += 1

            print(
                f"\n[OCR WORKER ERROR] {e}"
            )

            try:
                ocr_queue.task_done()
            except Exception:
                pass


# ============================================================
# STABLE PLATE
# ============================================================

def get_stable_plate():

    with ocr_lock:

        history = list(
            ocr_history
        )

    if not history:

        return (
            "",
            0.0,
            0,
            0.0
        )

    readings = [
        item["plate"]
        for item in history
        if item.get("plate")
    ]

    if not readings:

        return (
            "",
            0.0,
            0,
            0.0
        )

    # Exact vote
    counter = Counter(
        readings
    )

    candidate, votes = (
        counter.most_common(1)[0]
    )

    confidence_values = [

        item["confidence"]

        for item in history

        if item["plate"] == candidate
    ]

    if confidence_values:

        avg_confidence = (
            sum(confidence_values)
            /
            len(confidence_values)
        )

    else:

        avg_confidence = 0.0

    similarities = [

        fuzzy_similarity(
            candidate,
            plate
        )

        for plate in readings
    ]

    similarity = (

        sum(similarities)
        /
        len(similarities)

        if similarities
        else 0.0
    )

    return (
        candidate,
        avg_confidence,
        votes,
        similarity
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    total_frames,
    detection_runs,
    total_detections,
    average_fps,
    camera_fps
):

    stable_plate, stable_conf, stable_votes, similarity = (
        get_stable_plate()
    )

    history_file = os.path.join(
        OUTPUT_DIR,
        "ocr_history.json"
    )

    clusters_file = os.path.join(
        OUTPUT_DIR,
        "ocr_stability_clusters.json"
    )

    final_file = os.path.join(
        OUTPUT_DIR,
        "final_anpr_result.json"
    )

    # --------------------------------------------------------
    # OCR HISTORY
    # --------------------------------------------------------

    with ocr_lock:

        history_copy = list(
            ocr_history
        )

    with open(
        history_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            history_copy,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # STABILITY
    # --------------------------------------------------------

    stability_data = {

        "stable_plate":
            stable_plate,

        "stable_confidence":
            round(
                stable_conf,
                2
            ),

        "stable_votes":
            stable_votes,

        "fuzzy_similarity":
            round(
                similarity * 100,
                2
            ),

        "ocr_readings":
            len(history_copy)
    }

    with open(
        clusters_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            stability_data,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    final_result = {

        "phase":
            "6.5",

        "target_processing_fps":
            30,

        "total_frames":
            total_frames,

        "detection_runs":
            detection_runs,

        "total_yolo_detections":
            total_detections,

        "ocr_attempts":
            ocr_attempts,

        "ocr_successes":
            ocr_successes,

        "ocr_errors":
            ocr_errors,

        "camera_fps":
            round(
                camera_fps,
                2
            ),

        "processing_fps":
            round(
                average_fps,
                2
            ),

        "stable_plate":
            stable_plate,

        "stable_confidence":
            round(
                stable_conf,
                2
            ),

        "stable_votes":
            stable_votes,

        "fuzzy_similarity":
            round(
                similarity * 100,
                2
            ),

        "ocr_readings":
            len(history_copy),

        "target_30fps_achieved":
            average_fps >= 30
    }

    with open(
        final_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_result,
            f,
            indent=4
        )

    return final_result


# ============================================================
# MAIN
# ============================================================

def main():

    # IMPORTANT:
    # These are required because main() modifies them
    # when the user presses R.

    global latest_ocr_text
    global latest_ocr_confidence
    global latest_ocr_engine

    print("=" * 70)
    print(
        "AEGISVISION - ANPR"
    )
    print(
        "PHASE 6.5 - HIGH SPEED REAL-TIME ANPR"
    )
    print(
        "TARGET: 30+ PROCESSING FPS"
    )
    print(
        "YOLO + ASYNC PADDLEOCR"
    )
    print(
        "FAST DETECTION + PLATE TRACKING"
    )
    print("=" * 70)

    # ========================================================
    # LOAD YOLO
    # ========================================================

    print(
        "\nLoading YOLO plate detector..."
    )

    try:

        model = YOLO(
            MODEL_PATH
        )

        print(
            "YOLO detector loaded successfully."
        )

    except Exception as e:

        print(
            "\nERROR: Could not load YOLO model."
        )

        print(e)

        return

    # ========================================================
    # TESSERACT
    # ========================================================

    print(
        "\nChecking Tesseract OCR..."
    )

    try:

        if not os.path.exists(
            TESSERACT_PATH
        ):

            raise FileNotFoundError(
                TESSERACT_PATH
            )

        version = (
            pytesseract
            .get_tesseract_version()
        )

        print(
            "Tesseract executable found:"
        )

        print(
            TESSERACT_PATH
        )

        print(
            "Tesseract loaded successfully."
        )

        print(
            f"Version: {version}"
        )

    except Exception as e:

        print(
            "Tesseract unavailable."
        )

        print(e)

    # ========================================================
    # START OCR THREAD
    # ========================================================

    print(
        "\nStarting asynchronous OCR worker..."
    )

    ocr_thread = threading.Thread(
        target=ocr_worker,
        daemon=True
    )

    ocr_thread.start()

    # ========================================================
    # OPEN WEBCAM
    # ========================================================

    print(
        "\nOpening webcam..."
    )

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():

        print(
            "DirectShow failed."
        )

        print(
            "Trying default camera backend..."
        )

        cap = cv2.VideoCapture(
            CAMERA_INDEX
        )

    if not cap.isOpened():

        print(
            "ERROR: Could not open webcam."
        )

        ocr_stop_event.set()

        return

    # Camera settings
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
        TARGET_CAMERA_FPS
    )

    try:

        cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

    except Exception:
        pass

    actual_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    actual_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    camera_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    print(
        "\nCamera opened successfully."
    )

    print(
        "-" * 70
    )

    print(
        f"Resolution : "
        f"{actual_width} x {actual_height}"
    )

    print(
        f"Requested FPS : "
        f"{TARGET_CAMERA_FPS}"
    )

    print(
        f"Camera FPS    : "
        f"{camera_fps:.2f}"
    )

    print(
        "-" * 70
    )

    print(
        "\nPHASE 6.5 HIGH-SPEED ANPR STARTED"
    )

    print(
        "\nControls:"
    )

    print(
        "  Q / ESC = Quit"
    )

    print(
        "  S       = Save current frame"
    )

    print(
        "  R       = Reset OCR history"
    )

    # ========================================================
    # TRACKING
    # ========================================================

    current_box = None

    current_confidence = 0.0

    missed_frames = 0

    # ========================================================
    # COUNTERS
    # ========================================================

    total_frames = 0

    detection_runs = 0

    total_detections = 0

    frame_times = deque(
        maxlen=60
    )

    last_ocr_submitted_frame = (
        -OCR_INTERVAL
    )

    # ========================================================
    # MAIN LOOP
    # ========================================================

    try:

        while True:

            loop_start = (
                time.perf_counter()
            )

            ret, frame = (
                cap.read()
            )

            if not ret:

                print(
                    "\nCamera frame read failed."
                )

                break

            total_frames += 1

            # =================================================
            # YOLO DETECTION
            # =================================================

            run_detection = (

                current_box is None

                or

                total_frames %
                DETECTION_INTERVAL == 0
            )

            if run_detection:

                detection_runs += 1

                # Resize for faster YOLO
                small_frame = cv2.resize(
                    frame,
                    (
                        DETECTION_WIDTH,
                        DETECTION_HEIGHT
                    ),
                    interpolation=cv2.INTER_LINEAR
                )

                try:

                    results = model.predict(

                        small_frame,

                        conf=YOLO_CONFIDENCE,

                        verbose=False,

                        device="cpu",

                        imgsz=DETECTION_WIDTH
                    )

                    boxes = []

                    if results:

                        result = results[0]

                        if (
                            result.boxes
                            is not None
                        ):

                            for box in (
                                result.boxes
                            ):

                                xyxy = (
                                    box.xyxy[
                                        0
                                    ]
                                    .cpu()
                                    .numpy()
                                )

                                x1, y1, x2, y2 = (
                                    xyxy
                                )

                                conf = float(
                                    box.conf[
                                        0
                                    ]
                                    .cpu()
                                    .item()
                                )

                                # Scale to camera resolution
                                scale_x = (
                                    actual_width
                                    /
                                    DETECTION_WIDTH
                                )

                                scale_y = (
                                    actual_height
                                    /
                                    DETECTION_HEIGHT
                                )

                                x1 = int(
                                    x1 *
                                    scale_x
                                )

                                y1 = int(
                                    y1 *
                                    scale_y
                                )

                                x2 = int(
                                    x2 *
                                    scale_x
                                )

                                y2 = int(
                                    y2 *
                                    scale_y
                                )

                                x1 = max(
                                    0,
                                    min(
                                        x1,
                                        actual_width
                                        - 1
                                    )
                                )

                                y1 = max(
                                    0,
                                    min(
                                        y1,
                                        actual_height
                                        - 1
                                    )
                                )

                                x2 = max(
                                    0,
                                    min(
                                        x2,
                                        actual_width
                                        - 1
                                    )
                                )

                                y2 = max(
                                    0,
                                    min(
                                        y2,
                                        actual_height
                                        - 1
                                    )
                                )

                                if (
                                    x2 > x1
                                    and
                                    y2 > y1
                                ):

                                    boxes.append(
                                        (
                                            (
                                                x1,
                                                y1,
                                                x2,
                                                y2
                                            ),
                                            conf
                                        )
                                    )

                    # =================================================
                    # BEST PLATE
                    # =================================================

                    if boxes:

                        best_box, best_conf = (
                            max(
                                boxes,
                                key=lambda x:
                                x[1]
                            )
                        )

                        current_box = (
                            best_box
                        )

                        current_confidence = (
                            best_conf
                        )

                        missed_frames = 0

                        total_detections += 1

                    else:

                        missed_frames += 1

                except Exception as e:

                    print(
                        f"\nYOLO error: {e}"
                    )

                    missed_frames += 1

            # =================================================
            # KEEP TRACKING
            # =================================================

            else:

                # No YOLO call on this frame.
                # Keep previous bounding box.
                pass

            # =================================================
            # REMOVE LOST TRACK
            # =================================================

            if (
                missed_frames >
                MAX_MISSES
            ):

                current_box = None

                current_confidence = 0.0

                missed_frames = 0

            # =================================================
            # OCR SUBMISSION
            # =================================================

            if current_box is not None:

                x1, y1, x2, y2 = (
                    current_box
                )

                plate_width = (
                    x2 - x1
                )

                plate_height = (
                    y2 - y1
                )

                if (
                    plate_width >=
                    MIN_PLATE_WIDTH
                    and
                    plate_height >=
                    MIN_PLATE_HEIGHT
                ):

                    if (
                        total_frames
                        -
                        last_ocr_submitted_frame
                        >= OCR_INTERVAL
                    ):

                        crop = frame[
                            y1:y2,
                            x1:x2
                        ].copy()

                        try:

                            ocr_queue.put_nowait(
                                (
                                    crop,
                                    total_frames
                                )
                            )

                            last_ocr_submitted_frame = (
                                total_frames
                            )

                        except queue.Full:

                            # OCR is still processing.
                            # Skip this request.
                            pass

            # =================================================
            # DISPLAY
            # =================================================

            display = frame.copy()

            if current_box is not None:

                x1, y1, x2, y2 = (
                    current_box
                )

                cv2.rectangle(

                    display,

                    (
                        x1,
                        y1
                    ),

                    (
                        x2,
                        y2
                    ),

                    (0, 255, 0),

                    2
                )

                cv2.putText(

                    display,

                    (
                        "Detector: "
                        f"{current_confidence * 100:.1f}%"
                    ),

                    (
                        x1,
                        max(
                            20,
                            y1 - 10
                        )
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.55,

                    (0, 255, 0),

                    2
                )

            # =================================================
            # LATEST OCR
            # =================================================

            with ocr_lock:

                plate_text = (
                    latest_ocr_text
                )

                plate_confidence = (
                    latest_ocr_confidence
                )

                plate_engine = (
                    latest_ocr_engine
                )

            if plate_text:

                cv2.putText(

                    display,

                    f"PLATE: {plate_text}",

                    (
                        10,
                        35
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.9,

                    (0, 255, 0),

                    2
                )

                cv2.putText(

                    display,

                    (
                        f"OCR: "
                        f"{plate_confidence:.1f}% "
                        f"({plate_engine})"
                    ),

                    (
                        10,
                        65
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.55,

                    (0, 255, 0),

                    2
                )

            else:

                cv2.putText(

                    display,

                    "PLATE: Waiting for OCR...",

                    (
                        10,
                        35
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.7,

                    (0, 255, 255),

                    2
                )

            # =================================================
            # STABLE RESULT
            # =================================================

            (
                stable_plate,
                stable_conf,
                stable_votes,
                similarity
            ) = get_stable_plate()

            if stable_plate:

                cv2.putText(

                    display,

                    (
                        f"STABLE: "
                        f"{stable_plate} "
                        f"({stable_votes} votes)"
                    ),

                    (
                        10,
                        95
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.65,

                    (255, 255, 0),

                    2
                )

            # =================================================
            # FPS
            # =================================================

            loop_end = (
                time.perf_counter()
            )

            processing_time = (
                loop_end -
                loop_start
            )

            frame_times.append(
                processing_time
            )

            if frame_times:

                average_time = (
                    sum(frame_times)
                    /
                    len(frame_times)
                )

                processing_fps = (

                    1.0 /
                    average_time

                    if average_time > 0
                    else 0.0
                )

            else:

                processing_fps = 0.0

            # FPS color
            if processing_fps >= 30:

                fps_color = (
                    0,
                    255,
                    0
                )

            else:

                fps_color = (
                    0,
                    165,
                    255
                )

            cv2.putText(

                display,

                (
                    f"Processing FPS: "
                    f"{processing_fps:.1f}"
                ),

                (
                    10,
                    actual_height - 45
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.65,

                fps_color,

                2
            )

            cv2.putText(

                display,

                (
                    f"Camera FPS: "
                    f"{camera_fps:.1f}"
                ),

                (
                    10,
                    actual_height - 20
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (255, 255, 255),

                2
            )

            # =================================================
            # SHOW
            # =================================================

            cv2.imshow(

                "AEGISVISION - "
                "ANPR Phase 6.5",

                display
            )

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            # Quit
            if (
                key == ord("q")
                or
                key == 27
            ):

                break

            # Save frame
            elif key == ord("s"):

                save_path = os.path.join(

                    OUTPUT_DIR,

                    f"frame_{total_frames}.jpg"
                )

                cv2.imwrite(

                    save_path,

                    display
                )

                print(
                    f"\nFrame saved: "
                    f"{save_path}"
                )

            # Reset OCR
            elif key == ord("r"):

                with ocr_lock:

                    ocr_history.clear()

                    latest_ocr_text = ""

                    latest_ocr_confidence = (
                        0.0
                    )

                    latest_ocr_engine = ""

                print(
                    "\nOCR history reset."
                )

    except KeyboardInterrupt:

        print(
            "\nInterrupted by user."
        )

    finally:

        print(
            "\nStopping camera..."
        )

        cap.release()

        cv2.destroyAllWindows()

        ocr_stop_event.set()

        try:

            ocr_queue.put_nowait(
                None
            )

        except queue.Full:

            pass

        ocr_thread.join(
            timeout=3
        )

    # ========================================================
    # FINAL FPS
    # ========================================================

    if frame_times:

        average_time = (
            sum(frame_times)
            /
            len(frame_times)
        )

        average_fps = (

            1.0 /
            average_time

            if average_time > 0
            else 0.0
        )

    else:

        average_fps = 0.0

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    final_result = save_results(

        total_frames,

        detection_runs,

        total_detections,

        average_fps,

        camera_fps
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n")

    print("=" * 70)

    print(
        "PHASE 6.5 TEST RESULT"
    )

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
        f"{camera_fps:.2f}"
    )

    print(
        f"Average processing FPS : "
        f"{average_fps:.2f}"
    )

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

    (
        stable_plate,
        stable_conf,
        stable_votes,
        similarity
    ) = get_stable_plate()

    print()

    print(
        f"Stable plate           : "
        f"{stable_plate}"
    )

    print(
        f"Stable confidence      : "
        f"{stable_conf:.2f}%"
    )

    print(
        f"Stable votes           : "
        f"{stable_votes}"
    )

    print(
        f"Fuzzy similarity       : "
        f"{similarity * 100:.2f}%"
    )

    with ocr_lock:

        reading_count = len(
            ocr_history
        )

    print(
        f"OCR readings collected : "
        f"{reading_count}"
    )

    print()

    print(
        "Output directory:"
    )

    print(
        os.path.abspath(
            OUTPUT_DIR
        )
    )

    print()

    print(
        "OCR history:"
    )

    print(
        os.path.abspath(
            os.path.join(
                OUTPUT_DIR,
                "ocr_history.json"
            )
        )
    )

    print()

    print(
        "Stability clusters:"
    )

    print(
        os.path.abspath(
            os.path.join(
                OUTPUT_DIR,
                "ocr_stability_clusters.json"
            )
        )
    )

    print()

    print(
        "Final ANPR result:"
    )

    print(
        os.path.abspath(
            os.path.join(
                OUTPUT_DIR,
                "final_anpr_result.json"
            )
        )
    )

    print("=" * 70)

    # ========================================================
    # 30 FPS TARGET
    # ========================================================

    if average_fps >= 30:

        print(
            "\n🚀 30+ FPS TARGET ACHIEVED!"
        )

        print(
            f"Processing FPS = "
            f"{average_fps:.2f}"
        )

        print(
            "\nPHASE 6.5 PASSED."
        )

    else:

        print(
            "\n⚠ 30 FPS TARGET NOT YET ACHIEVED."
        )

        print(
            f"Current processing FPS = "
            f"{average_fps:.2f}"
        )

        print(
            "\nPHASE 6.5 IS NOT COMPLETE YET."
        )

        print(
            "Send the complete terminal output "
            "for the next optimization."
        )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()