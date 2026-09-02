"""
======================================================================
AEGISVISION - ANPR
PHASE 6.4 - REAL-TIME ANPR PERFORMANCE OPTIMIZATION
======================================================================

Features:
    - 60 FPS webcam capture
    - YOLO number plate detection
    - Intelligent frame skipping
    - Simple plate tracking using IoU
    - PaddleOCR primary OCR
    - Tesseract fallback
    - OCR result stabilization
    - Fuzzy multi-frame voting
    - OCR scheduling
    - Performance monitoring
    - Plate crop saving
    - JSON result generation

Controls:
    Q / ESC : Exit
    S       : Save current frame
    R       : Reset OCR history
======================================================================
"""

import os
import re
import cv2
import json
import time
import difflib
import traceback
from collections import Counter, deque

import numpy as np

# ----------------------------------------------------------------------
# OPTIONAL TESSERACT
# ----------------------------------------------------------------------

try:
    import pytesseract
    from pytesseract import Output

    TESSERACT_AVAILABLE = True

except Exception:
    pytesseract = None
    Output = None
    TESSERACT_AVAILABLE = False


# ======================================================================
# CONFIGURATION
# ======================================================================

MODEL_PATH = "models/plate_detector.pt"

OUTPUT_DIR = "output/phase6_4"
CROP_DIR = os.path.join(OUTPUT_DIR, "plate_crops")

CAMERA_INDEX = 0

# Camera
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 60

# YOLO
DETECTOR_CONFIDENCE = 0.25

# Process YOLO only every N frames
DETECTION_INTERVAL = 3

# Minimum detected plate size
MIN_PLATE_WIDTH = 40
MIN_PLATE_HEIGHT = 15

# OCR
OCR_INTERVAL = 12

# OCR confidence
MIN_OCR_CONFIDENCE = 45.0

# Stabilization
HISTORY_SIZE = 20
MIN_STABLE_READINGS = 3
FUZZY_THRESHOLD = 0.68

# Tracking
TRACK_IOU_THRESHOLD = 0.25
MAX_TRACK_MISSES = 15

# OCR preprocessing
MAX_PADDLE_VARIANTS = 2
MAX_TESSERACT_VARIANTS = 3

# Display
DISPLAY_SCALE = 1.0


# ======================================================================
# CREATE OUTPUT DIRECTORIES
# ======================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)


# ======================================================================
# UTILITY FUNCTIONS
# ======================================================================

def clean_text(text):
    """
    Clean OCR output.
    Keep only A-Z and 0-9.
    """

    if text is None:
        return ""

    text = str(text).upper()

    text = text.replace(" ", "")
    text = text.replace("-", "")
    text = text.replace("_", "")
    text = text.replace(".", "")
    text = text.replace(":", "")

    text = re.sub(r"[^A-Z0-9]", "", text)

    return text


def normalize_for_comparison(text):
    """
    Normalize OCR text for fuzzy comparison.

    This does NOT blindly replace O/0 or I/1 because that can
    incorrectly change a genuine plate.
    """

    text = clean_text(text)

    return text


def fuzzy_similarity(a, b):
    """
    Return similarity between two OCR readings.
    """

    a = normalize_for_comparison(a)
    b = normalize_for_comparison(b)

    if not a or not b:
        return 0.0

    return difflib.SequenceMatcher(None, a, b).ratio()


def plate_iou(box1, box2):
    """
    Calculate Intersection over Union.
    """

    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2

    ix1 = max(x1, a1)
    iy1 = max(y1, b1)
    ix2 = min(x2, a2)
    iy2 = min(y2, b2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    intersection = iw * ih

    area1 = max(0, x2 - x1) * max(0, y2 - y1)
    area2 = max(0, a2 - a1) * max(0, b2 - b1)

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def clamp_box(box, width, height):
    """
    Keep bounding box inside image.
    """

    x1, y1, x2, y2 = box

    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width - 1))
    y2 = max(0, min(int(y2), height - 1))

    return x1, y1, x2, y2


# ======================================================================
# OCR PREPROCESSING
# ======================================================================

def create_ocr_variants(crop):
    """
    Create lightweight OCR preprocessing variants.
    """

    variants = []

    if crop is None or crop.size == 0:
        return variants

    # --------------------------------------------------------------
    # Resize
    # --------------------------------------------------------------

    h, w = crop.shape[:2]

    scale = 3.0

    resized = cv2.resize(
        crop,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    variants.append(("original", resized))

    # --------------------------------------------------------------
    # Grayscale + threshold
    # --------------------------------------------------------------

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    # OTSU
    _, otsu = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    variants.append(("threshold", cv2.cvtColor(
        otsu,
        cv2.COLOR_GRAY2BGR
    )))

    # --------------------------------------------------------------
    # Sharpen
    # --------------------------------------------------------------

    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    sharpened = cv2.filter2D(resized, -1, kernel)

    variants.append(("sharp", sharpened))

    return variants


# ======================================================================
# PADDLE OCR RESULT EXTRACTION
# ======================================================================

def extract_paddle_result(result):
    """
    Extract text and confidence from different PaddleOCR result formats.
    """

    texts = []
    scores = []

    try:

        # ----------------------------------------------------------
        # New PaddleOCR result object
        # ----------------------------------------------------------

        data = None

        if hasattr(result, "json"):

            data = result.json

            if callable(data):
                data = data()

        elif isinstance(result, dict):
            data = result

        # ----------------------------------------------------------
        # Sometimes JSON is nested
        # ----------------------------------------------------------

        if isinstance(data, str):

            try:
                data = json.loads(data)
            except Exception:
                data = None

        if isinstance(data, dict):

            # Common PaddleOCR structure
            if "res" in data and isinstance(data["res"], dict):
                data = data["res"]

            rec_texts = data.get("rec_texts", [])
            rec_scores = data.get("rec_scores", [])

            if rec_texts:

                texts = [
                    clean_text(x)
                    for x in rec_texts
                ]

                scores = [
                    float(x) * 100
                    for x in rec_scores
                ]

                return texts, scores

        # ----------------------------------------------------------
        # Attribute based
        # ----------------------------------------------------------

        if hasattr(result, "rec_texts"):

            rec_texts = result.rec_texts
            rec_scores = getattr(result, "rec_scores", [])

            texts = [
                clean_text(x)
                for x in rec_texts
            ]

            scores = [
                float(x) * 100
                for x in rec_scores
            ]

            return texts, scores

    except Exception:
        pass

    return [], []


# ======================================================================
# PADDLE OCR
# ======================================================================

def run_paddle_ocr(ocr, crop):
    """
    Run PaddleOCR on selected preprocessing variants.
    """

    if ocr is None:
        return None, 0.0

    variants = create_ocr_variants(crop)

    variants = variants[:MAX_PADDLE_VARIANTS]

    best_text = ""
    best_conf = 0.0

    for variant_name, image in variants:

        try:

            # PaddleOCR expects 3-channel BGR image
            if len(image.shape) == 2:

                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_GRAY2BGR
                )

            result = ocr.predict(input=image)

            if result is None:
                continue

            for item in result:

                texts, scores = extract_paddle_result(item)

                for text, score in zip(texts, scores):

                    text = clean_text(text)

                    if not text:
                        continue

                    if score > best_conf:

                        best_text = text
                        best_conf = score

        except Exception as exc:

            raise RuntimeError(
                f"PaddleOCR runtime error: {exc}"
            )

    if best_text:
        return best_text, best_conf

    return None, 0.0


# ======================================================================
# TESSERACT OCR
# ======================================================================

def run_tesseract_ocr(crop):
    """
    Tesseract fallback OCR.
    """

    if not TESSERACT_AVAILABLE:
        return None, 0.0

    if pytesseract is None:
        return None, 0.0

    variants = create_ocr_variants(crop)

    variants = variants[:MAX_TESSERACT_VARIANTS]

    best_text = ""
    best_conf = 0.0

    configs = [
        "--psm 7",
        "--psm 8",
        "--psm 13"
    ]

    for index, (variant_name, image) in enumerate(variants):

        if index >= len(configs):
            break

        try:

            data = pytesseract.image_to_data(
                image,
                config=(
                    configs[index]
                    + " -c tessedit_char_whitelist="
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                ),
                output_type=Output.DICT
            )

            collected = []
            confidence_values = []

            for text, conf in zip(
                data["text"],
                data["conf"]
            ):

                text = clean_text(text)

                try:
                    conf = float(conf)
                except Exception:
                    conf = 0.0

                if text:

                    collected.append(text)

                    if conf >= 0:
                        confidence_values.append(conf)

            if collected:

                final_text = "".join(collected)

                if confidence_values:
                    final_conf = sum(
                        confidence_values
                    ) / len(confidence_values)
                else:
                    final_conf = 0.0

                if final_conf > best_conf:

                    best_text = final_text
                    best_conf = final_conf

        except Exception:
            continue

    if best_text:
        return best_text, best_conf

    return None, 0.0


# ======================================================================
# OCR STABILIZER
# ======================================================================

class OCRStabilizer:

    def __init__(self):

        self.history = deque(
            maxlen=HISTORY_SIZE
        )

        self.stable_plate = None
        self.stable_confidence = 0.0
        self.stable_votes = 0
        self.fuzzy_similarity = 0.0

    def reset(self):

        self.history.clear()

        self.stable_plate = None
        self.stable_confidence = 0.0
        self.stable_votes = 0
        self.fuzzy_similarity = 0.0

    def add(self, text, confidence):

        text = clean_text(text)

        if not text:
            return

        if confidence < MIN_OCR_CONFIDENCE:
            return

        self.history.append({
            "plate": text,
            "confidence": float(confidence),
            "timestamp": time.time()
        })

        self.calculate_stability()

    def calculate_stability(self):

        if not self.history:
            return

        readings = [
            item["plate"]
            for item in self.history
        ]

        # ----------------------------------------------------------
        # Exact votes
        # ----------------------------------------------------------

        counts = Counter(readings)

        best_exact, exact_votes = counts.most_common(1)[0]

        # ----------------------------------------------------------
        # Fuzzy grouping
        # ----------------------------------------------------------

        clusters = []

        for reading in readings:

            placed = False

            for cluster in clusters:

                similarity = fuzzy_similarity(
                    reading,
                    cluster["representative"]
                )

                if similarity >= FUZZY_THRESHOLD:

                    cluster["items"].append(reading)

                    cluster["similarities"].append(
                        similarity
                    )

                    placed = True
                    break

            if not placed:

                clusters.append({
                    "representative": reading,
                    "items": [reading],
                    "similarities": [1.0]
                })

        if not clusters:
            return

        best_cluster = max(
            clusters,
            key=lambda x: len(x["items"])
        )

        cluster_items = best_cluster["items"]

        # ----------------------------------------------------------
        # Select best representative
        # ----------------------------------------------------------

        representative_counts = Counter(
            cluster_items
        )

        representative = (
            representative_counts
            .most_common(1)[0][0]
        )

        cluster_confidences = []

        for item in self.history:

            if fuzzy_similarity(
                item["plate"],
                representative
            ) >= FUZZY_THRESHOLD:

                cluster_confidences.append(
                    item["confidence"]
                )

        if cluster_confidences:

            average_confidence = (
                sum(cluster_confidences)
                / len(cluster_confidences)
            )

        else:

            average_confidence = 0.0

        similarities = best_cluster["similarities"]

        average_similarity = (
            sum(similarities)
            / len(similarities)
            if similarities
            else 0.0
        )

        # ----------------------------------------------------------
        # Confirm stable result
        # ----------------------------------------------------------

        if len(cluster_items) >= MIN_STABLE_READINGS:

            self.stable_plate = representative

            self.stable_confidence = (
                average_confidence
            )

            self.stable_votes = len(
                cluster_items
            )

            self.fuzzy_similarity = (
                average_similarity
            )


# ======================================================================
# PLATE TRACKER
# ======================================================================

class PlateTracker:

    def __init__(self):

        self.box = None

        self.confidence = 0.0

        self.misses = 0

        self.detected = False

    def reset(self):

        self.box = None
        self.confidence = 0.0
        self.misses = 0
        self.detected = False

    def update(self, detections):

        # No detections
        if not detections:

            self.misses += 1

            if self.misses > MAX_TRACK_MISSES:

                self.reset()

            return None

        # First detection
        if self.box is None:

            best = max(
                detections,
                key=lambda x: x["confidence"]
            )

            self.box = best["box"]
            self.confidence = best["confidence"]

            self.misses = 0
            self.detected = True

            return best

        # Find closest IoU
        best_match = None
        best_iou = 0.0

        for detection in detections:

            iou = plate_iou(
                self.box,
                detection["box"]
            )

            if iou > best_iou:

                best_iou = iou
                best_match = detection

        if (
            best_match is not None
            and best_iou >= TRACK_IOU_THRESHOLD
        ):

            self.box = best_match["box"]

            self.confidence = (
                best_match["confidence"]
            )

            self.misses = 0
            self.detected = True

            return best_match

        # New detection if old track lost
        best = max(
            detections,
            key=lambda x: x["confidence"]
        )

        self.box = best["box"]

        self.confidence = best["confidence"]

        self.misses = 0
        self.detected = True

        return best


# ======================================================================
# YOLO DETECTOR
# ======================================================================

def load_yolo():

    print()
    print("Loading YOLO plate detector...")

    try:

        from ultralytics import YOLO

        model = YOLO(MODEL_PATH)

        print("YOLO detector loaded successfully.")

        return model

    except Exception as exc:

        print()
        print("ERROR: YOLO could not be loaded.")
        print(exc)

        raise


def detect_plates(model, frame):

    detections = []

    try:

        results = model.predict(
            source=frame,
            conf=DETECTOR_CONFIDENCE,
            verbose=False,
            imgsz=640
        )

        if not results:
            return detections

        result = results[0]

        if result.boxes is None:
            return detections

        boxes = result.boxes

        for i in range(len(boxes)):

            xyxy = boxes.xyxy[i].cpu().numpy()

            confidence = float(
                boxes.conf[i].cpu().item()
            )

            x1, y1, x2, y2 = map(
                int,
                xyxy
            )

            width = x2 - x1
            height = y2 - y1

            if (
                width < MIN_PLATE_WIDTH
                or height < MIN_PLATE_HEIGHT
            ):
                continue

            detections.append({
                "box": (x1, y1, x2, y2),
                "confidence": confidence
            })

    except Exception as exc:

        print()
        print(
            f"YOLO runtime error: {exc}"
        )

    return detections


# ======================================================================
# CAMERA SETUP
# ======================================================================

def open_camera():

    print()
    print("Opening webcam...")
    print()

    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    if not camera.isOpened():

        print(
            "DirectShow failed. Trying default backend..."
        )

        camera = cv2.VideoCapture(
            CAMERA_INDEX
        )

    if not camera.isOpened():

        raise RuntimeError(
            "Could not open webcam."
        )

    # --------------------------------------------------------------
    # MJPG
    # --------------------------------------------------------------

    camera.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(
            *"MJPG"
        )
    )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT
    )

    camera.set(
        cv2.CAP_PROP_FPS,
        TARGET_FPS
    )

    # Reduce buffering where supported
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

    actual_fps = camera.get(
        cv2.CAP_PROP_FPS
    )

    fourcc_value = int(
        camera.get(
            cv2.CAP_PROP_FOURCC
        )
    )

    codec = "".join(
        [
            chr(
                (fourcc_value >> 8 * i)
                & 0xFF
            )
            for i in range(4)
        ]
    )

    print(
        "Camera opened successfully."
    )

    print("-" * 70)

    print(
        f"Resolution : "
        f"{actual_width} x {actual_height}"
    )

    print(
        f"Requested FPS : {TARGET_FPS}"
    )

    print(
        f"Camera FPS    : {actual_fps:.2f}"
    )

    print(
        f"Video codec   : {codec}"
    )

    print("-" * 70)

    return camera, actual_fps


# ======================================================================
# PADDLE OCR SETUP
# ======================================================================

def load_paddleocr():

    print()
    print("Loading PaddleOCR...")

    try:

        from paddleocr import PaddleOCR

        ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
            enable_mkldnn=False,
            engine="paddle"
        )

        print(
            "PaddleOCR loaded successfully."
        )

        print(
            "PaddleOCR CPU mode enabled."
        )

        print(
            "PaddleOCR MKL-DNN explicitly disabled."
        )

        return ocr

    except TypeError:

        # Compatibility fallback
        print(
            "Using compatibility PaddleOCR configuration..."
        )

        from paddleocr import PaddleOCR

        ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
            enable_mkldnn=False
        )

        print(
            "PaddleOCR loaded successfully."
        )

        return ocr

    except Exception as exc:

        print()
        print(
            "WARNING: PaddleOCR could not be loaded."
        )

        print(exc)

        return None


# ======================================================================
# TESSERACT SETUP
# ======================================================================

def setup_tesseract():

    print()
    print("Checking Tesseract OCR...")

    if not TESSERACT_AVAILABLE:

        print(
            "Python pytesseract is not available."
        )

        return False

    possible_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
    ]

    executable = None

    for path in possible_paths:

        if os.path.exists(path):

            executable = path
            break

    if executable is None:

        print(
            "Tesseract executable not found."
        )

        return False

    pytesseract.pytesseract.tesseract_cmd = (
        executable
    )

    print(
        "Tesseract executable found:"
    )

    print(executable)

    try:

        version = pytesseract.get_tesseract_version()

        print(
            "Tesseract loaded successfully."
        )

        print(
            f"Version: {version}"
        )

        return True

    except Exception as exc:

        print(
            "Tesseract test failed:"
        )

        print(exc)

        return False


# ======================================================================
# DRAW DETECTION
# ======================================================================

def draw_interface(
    frame,
    tracked_detection,
    stabilizer,
    frame_number,
    camera_fps,
    measured_capture_fps,
    overall_fps,
    yolo_fps,
    ocr_fps,
    last_ocr_plate,
    last_ocr_conf,
    last_engine
):

    display = frame.copy()

    # --------------------------------------------------------------
    # Plate box
    # --------------------------------------------------------------

    if tracked_detection is not None:

        x1, y1, x2, y2 = (
            tracked_detection["box"]
        )

        detector_conf = (
            tracked_detection["confidence"]
            * 100
        )

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        label = (
            f"Plate Detector: "
            f"{detector_conf:.1f}%"
        )

        cv2.putText(
            display,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

    # --------------------------------------------------------------
    # Stable plate
    # --------------------------------------------------------------

    stable = stabilizer.stable_plate

    if stable:

        status = "STABLE"

        cv2.putText(
            display,
            f"PLATE: {stable}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            (
                f"Confidence: "
                f"{stabilizer.stable_confidence:.1f}%"
            ),
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            (
                f"Votes: "
                f"{stabilizer.stable_votes} | "
                f"Similarity: "
                f"{stabilizer.fuzzy_similarity * 100:.1f}%"
            ),
            (15, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

    else:

        status = "SEARCHING"

        cv2.putText(
            display,
            "PLATE: SEARCHING...",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2
        )

    # --------------------------------------------------------------
    # Last OCR
    # --------------------------------------------------------------

    if last_ocr_plate:

        cv2.putText(
            display,
            (
                f"Last OCR: "
                f"{last_ocr_plate} "
                f"({last_ocr_conf:.1f}%)"
            ),
            (15, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    cv2.putText(
        display,
        f"Engine: {last_engine}",
        (15, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------------
    # Performance panel
    # --------------------------------------------------------------

    panel_y = 175

    cv2.putText(
        display,
        f"Camera FPS : {camera_fps:.1f}",
        (15, panel_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"Capture FPS: {measured_capture_fps:.1f}",
        (15, panel_y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"Overall FPS: {overall_fps:.1f}",
        (15, panel_y + 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"YOLO FPS   : {yolo_fps:.1f}",
        (15, panel_y + 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"OCR FPS    : {ocr_fps:.1f}",
        (15, panel_y + 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"Frame: {frame_number}",
        (15, panel_y + 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        "Q: Exit | S: Save | R: Reset",
        (15, display.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )

    return display


# ======================================================================
# MAIN
# ======================================================================

def main():

    print()
    print("=" * 70)
    print("AEGISVISION - ANPR")
    print("PHASE 6.4 - REAL-TIME PERFORMANCE OPTIMIZATION")
    print("YOLO + PADDLEOCR + TESSERACT FALLBACK")
    print("INTELLIGENT FRAME SKIPPING")
    print("PLATE TRACKING")
    print("FUZZY MULTI-FRAME OCR VOTING")
    print("60 FPS WEBCAM MODE")
    print("=" * 70)

    # --------------------------------------------------------------
    # Load YOLO
    # --------------------------------------------------------------

    model = load_yolo()

    # --------------------------------------------------------------
    # Load PaddleOCR
    # --------------------------------------------------------------

    paddle_ocr = load_paddleocr()

    paddle_enabled = paddle_ocr is not None

    # --------------------------------------------------------------
    # Tesseract
    # --------------------------------------------------------------

    tesseract_enabled = setup_tesseract()

    # --------------------------------------------------------------
    # Camera
    # --------------------------------------------------------------

    camera, camera_fps = open_camera()

    # --------------------------------------------------------------
    # Components
    # --------------------------------------------------------------

    stabilizer = OCRStabilizer()

    tracker = PlateTracker()

    # --------------------------------------------------------------
    # Counters
    # --------------------------------------------------------------

    frame_number = 0

    total_frames = 0

    total_detections = 0

    yolo_runs = 0

    ocr_attempts = 0

    ocr_errors = 0

    paddle_successes = 0

    tesseract_successes = 0

    saved_frames = 0

    # --------------------------------------------------------------
    # Last values
    # --------------------------------------------------------------

    last_ocr_plate = None

    last_ocr_confidence = 0.0

    last_ocr_engine = "None"

    current_detection = None

    # --------------------------------------------------------------
    # FPS measurement
    # --------------------------------------------------------------

    program_start = time.perf_counter()

    last_frame_time = time.perf_counter()

    capture_times = deque(maxlen=30)

    overall_times = deque(maxlen=30)

    yolo_times = deque(maxlen=20)

    ocr_times = deque(maxlen=20)

    # --------------------------------------------------------------
    # JSON history
    # --------------------------------------------------------------

    ocr_history = []

    stability_history = []

    # --------------------------------------------------------------
    # Last OCR frame
    # --------------------------------------------------------------

    last_ocr_frame = -999999

    # --------------------------------------------------------------
    # Display
    # --------------------------------------------------------------

    window_name = (
        "AEGISVISION ANPR - PHASE 6.4"
    )

    print()
    print("=" * 70)
    print("REAL-TIME OPTIMIZED ANPR STARTED")
    print("=" * 70)
    print("Q / ESC = Exit")
    print("S       = Save current frame")
    print("R       = Reset OCR history")
    print("=" * 70)
    print()

    try:

        while True:

            loop_start = time.perf_counter()

            # ======================================================
            # READ FRAME
            # ======================================================

            ret, frame = camera.read()

            if not ret:

                print(
                    "WARNING: Failed to read webcam frame."
                )

                continue

            frame_number += 1
            total_frames += 1

            # ------------------------------------------------------
            # Capture FPS
            # ------------------------------------------------------

            now = time.perf_counter()

            delta = now - last_frame_time

            if delta > 0:

                capture_times.append(
                    1.0 / delta
                )

            last_frame_time = now

            # ======================================================
            # YOLO DETECTION
            # ======================================================

            if (
                frame_number
                % DETECTION_INTERVAL
                == 0
            ):

                yolo_start = time.perf_counter()

                detections = detect_plates(
                    model,
                    frame
                )

                yolo_elapsed = (
                    time.perf_counter()
                    - yolo_start
                )

                yolo_times.append(
                    yolo_elapsed
                )

                yolo_runs += 1

                total_detections += len(
                    detections
                )

                current_detection = (
                    tracker.update(
                        detections
                    )
                )

            else:

                # Keep previous tracked plate
                if tracker.box is not None:

                    current_detection = {
                        "box": tracker.box,
                        "confidence": tracker.confidence
                    }

            # ======================================================
            # OCR
            # ======================================================

            should_ocr = (
                current_detection is not None
                and (
                    frame_number
                    - last_ocr_frame
                    >= OCR_INTERVAL
                )
            )

            if should_ocr:

                x1, y1, x2, y2 = (
                    current_detection["box"]
                )

                h, w = frame.shape[:2]

                x1, y1, x2, y2 = clamp_box(
                    (x1, y1, x2, y2),
                    w,
                    h
                )

                crop = frame[
                    y1:y2,
                    x1:x2
                ]

                if (
                    crop is not None
                    and crop.size > 0
                ):

                    ocr_attempts += 1

                    last_ocr_frame = (
                        frame_number
                    )

                    ocr_start = time.perf_counter()

                    plate_text = None
                    plate_confidence = 0.0
                    engine = "None"

                    # --------------------------------------------------
                    # PaddleOCR
                    # --------------------------------------------------

                    if paddle_enabled:

                        try:

                            plate_text, plate_confidence = (
                                run_paddle_ocr(
                                    paddle_ocr,
                                    crop
                                )
                            )

                            if (
                                plate_text
                                and plate_confidence
                                >= MIN_OCR_CONFIDENCE
                            ):

                                engine = "PaddleOCR"

                                paddle_successes += 1

                        except Exception as exc:

                            ocr_errors += 1

                            print()
                            print(
                                "PaddleOCR runtime error."
                            )

                            print(exc)

                            # Disable Paddle after runtime failure
                            paddle_enabled = False

                            print(
                                "PaddleOCR disabled."
                            )

                            print(
                                "Switching to Tesseract fallback."
                            )

                    # --------------------------------------------------
                    # Tesseract fallback
                    # --------------------------------------------------

                    if (
                        not plate_text
                        and tesseract_enabled
                    ):

                        tess_text, tess_conf = (
                            run_tesseract_ocr(
                                crop
                            )
                        )

                        if (
                            tess_text
                            and tess_conf
                            >= MIN_OCR_CONFIDENCE
                        ):

                            plate_text = tess_text

                            plate_confidence = (
                                tess_conf
                            )

                            engine = "Tesseract"

                            tesseract_successes += 1

                    # --------------------------------------------------
                    # OCR timing
                    # --------------------------------------------------

                    ocr_elapsed = (
                        time.perf_counter()
                        - ocr_start
                    )

                    ocr_times.append(
                        ocr_elapsed
                    )

                    # --------------------------------------------------
                    # Save OCR result
                    # --------------------------------------------------

                    if plate_text:

                        plate_text = clean_text(
                            plate_text
                        )

                        last_ocr_plate = (
                            plate_text
                        )

                        last_ocr_confidence = (
                            plate_confidence
                        )

                        last_ocr_engine = (
                            engine
                        )

                        stabilizer.add(
                            plate_text,
                            plate_confidence
                        )

                        ocr_history.append({
                            "frame": frame_number,
                            "plate": plate_text,
                            "confidence": round(
                                plate_confidence,
                                2
                            ),
                            "engine": engine,
                            "timestamp": time.time()
                        })

                        stability_history.append({
                            "frame": frame_number,
                            "stable_plate": (
                                stabilizer.stable_plate
                            ),
                            "stable_confidence": round(
                                stabilizer.stable_confidence,
                                2
                            ),
                            "stable_votes": (
                                stabilizer.stable_votes
                            ),
                            "fuzzy_similarity": round(
                                stabilizer.fuzzy_similarity,
                                4
                            )
                        })

                        print(
                            f"[Frame {frame_number}] "
                            f"Plate: {plate_text} | "
                            f"Detector: "
                            f"{current_detection['confidence'] * 100:.1f}% | "
                            f"OCR: "
                            f"{plate_confidence:.1f}% | "
                            f"Engine: {engine}"
                        )

            # ======================================================
            # PERFORMANCE
            # ======================================================

            loop_elapsed = (
                time.perf_counter()
                - loop_start
            )

            if loop_elapsed > 0:

                overall_times.append(
                    1.0 / loop_elapsed
                )

            # Camera reported FPS
            display_camera_fps = (
                camera_fps
            )

            # Measured capture FPS
            if capture_times:

                measured_capture_fps = (
                    sum(capture_times)
                    / len(capture_times)
                )

            else:

                measured_capture_fps = 0.0

            # Overall processing FPS
            if overall_times:

                overall_fps = (
                    sum(overall_times)
                    / len(overall_times)
                )

            else:

                overall_fps = 0.0

            # YOLO FPS
            if yolo_times:

                avg_yolo_time = (
                    sum(yolo_times)
                    / len(yolo_times)
                )

                yolo_fps = (
                    1.0 / avg_yolo_time
                    if avg_yolo_time > 0
                    else 0.0
                )

            else:

                yolo_fps = 0.0

            # OCR FPS
            if ocr_times:

                avg_ocr_time = (
                    sum(ocr_times)
                    / len(ocr_times)
                )

                ocr_fps = (
                    1.0 / avg_ocr_time
                    if avg_ocr_time > 0
                    else 0.0
                )

            else:

                ocr_fps = 0.0

            # ======================================================
            # DRAW UI
            # ======================================================

            display = draw_interface(
                frame,
                current_detection,
                stabilizer,
                frame_number,
                display_camera_fps,
                measured_capture_fps,
                overall_fps,
                yolo_fps,
                ocr_fps,
                last_ocr_plate,
                last_ocr_confidence,
                last_ocr_engine
            )

            cv2.imshow(
                window_name,
                display
            )

            # ======================================================
            # KEYBOARD
            # ======================================================

            key = cv2.waitKey(1) & 0xFF

            # ------------------------------------------------------
            # Exit
            # ------------------------------------------------------

            if key in (
                ord("q"),
                ord("Q"),
                27
            ):

                break

            # ------------------------------------------------------
            # Reset
            # ------------------------------------------------------

            elif key in (
                ord("r"),
                ord("R")
            ):

                stabilizer.reset()

                tracker.reset()

                current_detection = None

                last_ocr_plate = None

                last_ocr_confidence = 0.0

                last_ocr_engine = "None"

                print()
                print(
                    "OCR history and tracker reset."
                )
                print()

            # ------------------------------------------------------
            # Save frame
            # ------------------------------------------------------

            elif key in (
                ord("s"),
                ord("S")
            ):

                saved_frames += 1

                filename = os.path.join(
                    OUTPUT_DIR,
                    (
                        f"frame_"
                        f"{frame_number}_"
                        f"{saved_frames}.jpg"
                    )
                )

                cv2.imwrite(
                    filename,
                    frame
                )

                print()
                print(
                    f"Frame saved: {filename}"
                )
                print()

    except KeyboardInterrupt:

        print()
        print(
            "Stopped by keyboard interrupt."
        )

    except Exception as exc:

        print()
        print("=" * 70)
        print("UNEXPECTED ERROR")
        print("=" * 70)
        print(exc)
        traceback.print_exc()

    finally:

        # ==========================================================
        # RELEASE
        # ==========================================================

        camera.release()

        cv2.destroyAllWindows()

        # ==========================================================
        # FINAL PERFORMANCE
        # ==========================================================

        total_runtime = (
            time.perf_counter()
            - program_start
        )

        if total_runtime > 0:

            average_processing_fps = (
                total_frames
                / total_runtime
            )

        else:

            average_processing_fps = 0.0

        # ==========================================================
        # SAVE OCR HISTORY
        # ==========================================================

        history_path = os.path.join(
            OUTPUT_DIR,
            "ocr_history.json"
        )

        with open(
            history_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                ocr_history,
                file,
                indent=4
            )

        # ==========================================================
        # SAVE STABILITY CLUSTERS
        # ==========================================================

        clusters = []

        readings = [
            item["plate"]
            for item in ocr_history
        ]

        for reading in readings:

            matched = False

            for cluster in clusters:

                if (
                    fuzzy_similarity(
                        reading,
                        cluster["representative"]
                    )
                    >= FUZZY_THRESHOLD
                ):

                    cluster["readings"].append(
                        reading
                    )

                    matched = True
                    break

            if not matched:

                clusters.append({
                    "representative": reading,
                    "readings": [reading]
                })

        clusters_path = os.path.join(
            OUTPUT_DIR,
            "ocr_stability_clusters.json"
        )

        with open(
            clusters_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                clusters,
                file,
                indent=4
            )

        # ==========================================================
        # FINAL ANPR RESULT
        # ==========================================================

        final_result = {

            "project": "AEGISVISION",

            "module": "ANPR",

            "phase": "6.4",

            "status": "COMPLETED",

            "camera": {
                "index": CAMERA_INDEX,
                "width": FRAME_WIDTH,
                "height": FRAME_HEIGHT,
                "requested_fps": TARGET_FPS,
                "reported_fps": round(
                    camera_fps,
                    2
                )
            },

            "performance": {
                "total_frames": total_frames,
                "average_processing_fps": round(
                    average_processing_fps,
                    2
                ),
                "yolo_runs": yolo_runs,
                "ocr_attempts": ocr_attempts
            },

            "detection": {
                "total_yolo_detections": (
                    total_detections
                )
            },

            "ocr": {
                "paddleocr_successes": (
                    paddle_successes
                ),
                "tesseract_successes": (
                    tesseract_successes
                ),
                "ocr_errors": ocr_errors
            },

            "final_plate": (
                stabilizer.stable_plate
            ),

            "final_confidence": round(
                stabilizer.stable_confidence,
                2
            ),

            "stable_votes": (
                stabilizer.stable_votes
            ),

            "fuzzy_similarity": round(
                stabilizer.fuzzy_similarity,
                4
            ),

            "ocr_readings": len(
                ocr_history
            ),

            "configuration": {
                "detection_interval": (
                    DETECTION_INTERVAL
                ),
                "ocr_interval": (
                    OCR_INTERVAL
                ),
                "history_size": (
                    HISTORY_SIZE
                ),
                "fuzzy_threshold": (
                    FUZZY_THRESHOLD
                )
            },

            "timestamp": time.time()
        }

        final_path = os.path.join(
            OUTPUT_DIR,
            "final_anpr_result.json"
        )

        with open(
            final_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                final_result,
                file,
                indent=4
            )

        # ==========================================================
        # FINAL REPORT
        # ==========================================================

        print()
        print("=" * 70)
        print("PHASE 6.4 COMPLETED")
        print("=" * 70)

        print(
            f"Total frames processed : "
            f"{total_frames}"
        )

        print(
            f"Total YOLO detections  : "
            f"{total_detections}"
        )

        print(
            f"YOLO detection runs    : "
            f"{yolo_runs}"
        )

        print(
            f"OCR attempts           : "
            f"{ocr_attempts}"
        )

        print(
            f"OCR errors             : "
            f"{ocr_errors}"
        )

        print(
            f"PaddleOCR successes    : "
            f"{paddle_successes}"
        )

        print(
            f"Tesseract successes    : "
            f"{tesseract_successes}"
        )

        print(
            f"Camera FPS             : "
            f"{camera_fps:.2f}"
        )

        print(
            f"Average processing FPS : "
            f"{average_processing_fps:.2f}"
        )

        print(
            f"Last OCR plate         : "
            f"{last_ocr_plate}"
        )

        print(
            f"Last OCR confidence    : "
            f"{last_ocr_confidence:.2f}%"
        )

        print(
            f"Last OCR engine        : "
            f"{last_ocr_engine}"
        )

        print()

        print(
            f"Stable plate           : "
            f"{stabilizer.stable_plate}"
        )

        print(
            f"Stable confidence      : "
            f"{stabilizer.stable_confidence:.2f}%"
        )

        print(
            f"Stable votes           : "
            f"{stabilizer.stable_votes}"
        )

        print(
            f"Fuzzy similarity       : "
            f"{stabilizer.fuzzy_similarity * 100:.2f}%"
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
                history_path
            )
        )

        print()
        print("Stability clusters:")
        print(
            os.path.abspath(
                clusters_path
            )
        )

        print()
        print("Final ANPR result:")
        print(
            os.path.abspath(
                final_path
            )
        )

        print()
        print("=" * 70)
        print(
            "PHASE 6.4 PERFORMANCE OPTIMIZATION FINISHED."
        )
        print("=" * 70)


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()