# ================================================================
# AEGISVISION - ANPR
# PHASE 6.3 - ROBUST OCR RESULT STABILIZATION
#
# YOLO + PaddleOCR + Tesseract Fallback
# Fuzzy Multi-Frame OCR Voting
# 60 FPS Webcam Capture
# Real-Time FPS Monitoring
# ================================================================

import cv2
import os
import re
import json
import time
from collections import deque
from difflib import SequenceMatcher


# ================================================================
# CONFIGURATION
# ================================================================

MODEL_PATH = "models/plate_detector.pt"

OUTPUT_DIR = "output/phase6_3"
CROP_DIR = os.path.join(OUTPUT_DIR, "plate_crops")

CAMERA_INDEX = 0

# ------------------------------------------------
# 60 FPS CAMERA SETTINGS
# ------------------------------------------------

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 60

# ------------------------------------------------
# YOLO
# ------------------------------------------------

DETECTOR_CONFIDENCE = 0.25

MIN_PLATE_WIDTH = 40
MIN_PLATE_HEIGHT = 15

# ------------------------------------------------
# OCR
# ------------------------------------------------

OCR_INTERVAL = 8

HISTORY_SIZE = 15

MIN_OCR_CONFIDENCE = 45.0

MIN_STABLE_READINGS = 3

FUZZY_THRESHOLD = 0.68

MAX_PADDLE_VARIANTS = 3
MAX_TESSERACT_VARIANTS = 3


# ================================================================
# CREATE DIRECTORIES
# ================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)


# ================================================================
# GLOBAL VARIABLES
# ================================================================

ocr_history = deque(
    maxlen=HISTORY_SIZE
)

last_ocr_frame = -OCR_INTERVAL

last_plate_text = None
last_ocr_confidence = 0.0
last_ocr_engine = None

stable_plate = None
stable_confidence = 0.0
stable_votes = 0
stable_similarity = 0.0

total_frames = 0
total_detections = 0
ocr_attempts = 0
ocr_errors = 0

paddle_successes = 0
tesseract_successes = 0

ocr_runtime_errors = 0

paddle_ocr = None
PADDLE_ENABLED = False

tesseract_available = False


# ================================================================
# FPS VARIABLES
# ================================================================

fps_timer = time.time()

fps_frame_count = 0

processing_fps = 0.0

display_camera_fps = 0.0


# ================================================================
# IMPORT YOLO
# ================================================================

try:

    from ultralytics import YOLO

except ImportError:

    print()
    print("ERROR: ultralytics is not installed.")
    print()
    print("Run:")
    print("pip install ultralytics")
    print()

    raise SystemExit(1)


# ================================================================
# IMPORT PADDLEOCR
# ================================================================

try:

    from paddleocr import PaddleOCR

    PADDLE_IMPORT_AVAILABLE = True

except Exception as e:

    print()
    print("PaddleOCR import failed.")
    print("Reason:", e)
    print()

    PADDLE_IMPORT_AVAILABLE = False


# ================================================================
# IMPORT TESSERACT
# ================================================================

try:

    import pytesseract
    from pytesseract import Output

    TESSERACT_IMPORT_AVAILABLE = True

except Exception as e:

    print()
    print("Tesseract Python package unavailable.")
    print("Reason:", e)
    print()

    TESSERACT_IMPORT_AVAILABLE = False


# ================================================================
# NORMALIZE OCR TEXT
# ================================================================

def normalize_plate(text):

    if text is None:
        return ""

    text = str(text).upper()

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text


# ================================================================
# PLATE FORMAT SCORE
# ================================================================

def plate_format_score(text):

    text = normalize_plate(text)

    if not text:
        return 0.0

    # Typical Indian registration format
    full_pattern = (
        r"^[A-Z]{2}\d{1,2}"
        r"[A-Z]{1,3}\d{1,4}$"
    )

    if re.fullmatch(
        full_pattern,
        text
    ):

        return 1.0

    if len(text) >= 6:
        return 0.65

    if len(text) >= 4:
        return 0.35

    return 0.0


# ================================================================
# FUZZY SIMILARITY
# ================================================================

def fuzzy_similarity(a, b):

    a = normalize_plate(a)
    b = normalize_plate(b)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    # One OCR result contained inside another
    if a in b or b in a:

        shorter = min(
            len(a),
            len(b)
        )

        if shorter >= 5:
            return 0.88

        if shorter >= 4:
            return 0.80

    ratio = SequenceMatcher(
        None,
        a,
        b
    ).ratio()

    # Prefix compatibility
    prefix = 0

    for x, y in zip(a, b):

        if x == y:
            prefix += 1

        else:
            break

    min_length = min(
        len(a),
        len(b)
    )

    if min_length > 0:

        prefix_ratio = (
            prefix / min_length
        )

        if prefix_ratio >= 0.75:

            ratio = max(
                ratio,
                0.84
            )

    return ratio


# ================================================================
# IMAGE PREPROCESSING
# ================================================================

def create_ocr_variants(plate_image):

    if (
        plate_image is None
        or
        plate_image.size == 0
    ):

        return []

    # White border
    image = cv2.copyMakeBorder(
        plate_image,
        8,
        8,
        8,
        8,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255)
    )

    # 3x upscale
    upscaled = cv2.resize(
        image,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    variants = []

    # ------------------------------------------------
    # Variant 1
    # Original upscaled
    # ------------------------------------------------

    variants.append(
        upscaled
    )

    # ------------------------------------------------
    # Variant 2
    # CLAHE enhanced
    # ------------------------------------------------

    gray = cv2.cvtColor(
        upscaled,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        gray
    )

    enhanced_bgr = cv2.cvtColor(
        enhanced,
        cv2.COLOR_GRAY2BGR
    )

    variants.append(
        enhanced_bgr
    )

    # ------------------------------------------------
    # Variant 3
    # Adaptive threshold
    # ------------------------------------------------

    blurred = cv2.GaussianBlur(
        enhanced,
        (3, 3),
        0
    )

    threshold = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9
    )

    threshold_bgr = cv2.cvtColor(
        threshold,
        cv2.COLOR_GRAY2BGR
    )

    variants.append(
        threshold_bgr
    )

    return variants


# ================================================================
# EXTRACT PADDLEOCR RESULT
# ================================================================

def extract_paddle_result(result):

    texts = []
    scores = []

    try:

        # ------------------------------------------------
        # JSON property
        # ------------------------------------------------

        if hasattr(
            result,
            "json"
        ):

            data = result.json

            if callable(data):
                data = data()

            if isinstance(
                data,
                str
            ):

                data = json.loads(
                    data
                )

            if isinstance(
                data,
                dict
            ):

                if "rec_texts" in data:

                    texts.extend(
                        data.get(
                            "rec_texts",
                            []
                        )
                    )

                    scores.extend(
                        data.get(
                            "rec_scores",
                            []
                        )
                    )

                if (
                    "res" in data
                    and
                    isinstance(
                        data["res"],
                        dict
                    )
                ):

                    nested = data["res"]

                    texts.extend(
                        nested.get(
                            "rec_texts",
                            []
                        )
                    )

                    scores.extend(
                        nested.get(
                            "rec_scores",
                            []
                        )
                    )

        # ------------------------------------------------
        # Direct dictionary
        # ------------------------------------------------

        elif isinstance(
            result,
            dict
        ):

            if "rec_texts" in result:

                texts.extend(
                    result.get(
                        "rec_texts",
                        []
                    )
                )

                scores.extend(
                    result.get(
                        "rec_scores",
                        []
                    )
                )

            elif "res" in result:

                nested = result["res"]

                if isinstance(
                    nested,
                    dict
                ):

                    texts.extend(
                        nested.get(
                            "rec_texts",
                            []
                        )
                    )

                    scores.extend(
                        nested.get(
                            "rec_scores",
                            []
                        )
                    )

        # ------------------------------------------------
        # Attributes
        # ------------------------------------------------

        if hasattr(
            result,
            "rec_texts"
        ):

            try:

                texts.extend(
                    list(
                        result.rec_texts
                    )
                )

            except Exception:
                pass

        if hasattr(
            result,
            "rec_scores"
        ):

            try:

                scores.extend(
                    list(
                        result.rec_scores
                    )
                )

            except Exception:
                pass

    except Exception:
        pass

    # ------------------------------------------------
    # Clean results
    # ------------------------------------------------

    cleaned = []

    for i, text in enumerate(texts):

        text = normalize_plate(
            text
        )

        if not text:
            continue

        confidence = 0.0

        if i < len(scores):

            try:

                confidence = float(
                    scores[i]
                )

                if confidence <= 1.0:

                    confidence *= 100.0

            except Exception:

                confidence = 0.0

        cleaned.append(
            (
                text,
                confidence
            )
        )

    return cleaned


# ================================================================
# DISABLE PADDLEOCR
# ================================================================

def disable_paddleocr(reason):

    global PADDLE_ENABLED
    global paddle_ocr
    global ocr_runtime_errors

    ocr_runtime_errors += 1

    PADDLE_ENABLED = False

    paddle_ocr = None

    print()
    print("=" * 70)
    print("PADDLEOCR RUNTIME FAILURE")
    print("=" * 70)
    print(
        "PaddleOCR has been disabled "
        "for this run."
    )
    print(
        "Switching to Tesseract fallback."
    )
    print()
    print("Reason:")
    print(str(reason))
    print("=" * 70)
    print()


# ================================================================
# RUN PADDLEOCR
# ================================================================

def run_paddle_ocr(plate_image):

    global paddle_successes

    if not PADDLE_ENABLED:
        return []

    variants = create_ocr_variants(
        plate_image
    )

    all_results = []

    try:

        for image in variants[
            :MAX_PADDLE_VARIANTS
        ]:

            result_list = (
                paddle_ocr.predict(
                    input=image
                )
            )

            if result_list is None:
                continue

            if not isinstance(
                result_list,
                list
            ):

                result_list = [
                    result_list
                ]

            for result in result_list:

                extracted = (
                    extract_paddle_result(
                        result
                    )
                )

                all_results.extend(
                    extracted
                )

        if all_results:

            paddle_successes += 1

        return all_results

    except Exception as e:

        disable_paddleocr(e)

        return []


# ================================================================
# RUN TESSERACT
# ================================================================

def run_tesseract_ocr(plate_image):

    global tesseract_successes

    if not tesseract_available:
        return []

    if (
        plate_image is None
        or
        plate_image.size == 0
    ):

        return []

    try:

        variants = create_ocr_variants(
            plate_image
        )

        results = []

        psm_modes = [
            7,
            8,
            13
        ]

        for image in variants[
            :MAX_TESSERACT_VARIANTS
        ]:

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

            for psm in psm_modes:

                config = (
                    f"--psm {psm} "
                    "-c "
                    "tessedit_char_whitelist="
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                )

                data = (
                    pytesseract.image_to_data(
                        gray,
                        config=config,
                        output_type=Output.DICT
                    )
                )

                texts = data.get(
                    "text",
                    []
                )

                confidences = data.get(
                    "conf",
                    []
                )

                for i, text in enumerate(
                    texts
                ):

                    text = normalize_plate(
                        text
                    )

                    if not text:
                        continue

                    confidence = 0.0

                    if i < len(
                        confidences
                    ):

                        try:

                            confidence = float(
                                confidences[i]
                            )

                        except Exception:

                            confidence = 0.0

                    if confidence < 0:
                        continue

                    results.append(
                        (
                            text,
                            confidence
                        )
                    )

        # ------------------------------------------------
        # Filter candidates
        # ------------------------------------------------

        valid_results = []

        for text, confidence in results:

            if len(text) < 4:
                continue

            format_score = (
                plate_format_score(
                    text
                )
            )

            combined_score = (
                confidence * 0.75
                +
                format_score * 25.0
            )

            valid_results.append(
                (
                    text,
                    confidence,
                    combined_score
                )
            )

        if not valid_results:
            return []

        valid_results.sort(
            key=lambda x: x[2],
            reverse=True
        )

        final_results = []

        seen = set()

        for text, confidence, _ in (
            valid_results
        ):

            if text in seen:
                continue

            seen.add(text)

            final_results.append(
                (
                    text,
                    confidence
                )
            )

            if len(
                final_results
            ) >= 5:

                break

        if final_results:

            tesseract_successes += 1

        return final_results

    except Exception as e:

        print(
            "Tesseract OCR error:",
            e
        )

        return []


# ================================================================
# COMBINED OCR
# ================================================================

def run_combined_ocr(plate_image):

    # ------------------------------------------------
    # PaddleOCR
    # ------------------------------------------------

    if PADDLE_ENABLED:

        paddle_results = (
            run_paddle_ocr(
                plate_image
            )
        )

        useful = []

        for text, confidence in (
            paddle_results
        ):

            if len(text) >= 4:

                useful.append(
                    (
                        text,
                        confidence,
                        "PaddleOCR"
                    )
                )

        if useful:

            useful.sort(
                key=lambda x: x[1],
                reverse=True
            )

            return useful

    # ------------------------------------------------
    # Tesseract fallback
    # ------------------------------------------------

    tesseract_results = (
        run_tesseract_ocr(
            plate_image
        )
    )

    useful = []

    for text, confidence in (
        tesseract_results
    ):

        if len(text) >= 4:

            useful.append(
                (
                    text,
                    confidence,
                    "Tesseract"
                )
            )

    return useful


# ================================================================
# ADD OCR READING
# ================================================================

def add_ocr_reading(
    text,
    confidence,
    engine,
    frame_number
):

    text = normalize_plate(
        text
    )

    if not text:
        return

    if confidence < MIN_OCR_CONFIDENCE:
        return

    reading = {
        "text": text,
        "confidence": round(
            float(confidence),
            2
        ),
        "engine": engine,
        "frame": frame_number,
        "timestamp": time.time()
    }

    ocr_history.append(
        reading
    )


# ================================================================
# STABILIZE OCR
# ================================================================

def stabilize_plate():

    if not ocr_history:

        return (
            None,
            0.0,
            0,
            0.0,
            []
        )

    candidates = []

    for reading in ocr_history:

        text = reading["text"]

        if text not in candidates:

            candidates.append(
                text
            )

    clusters = []

    for candidate in candidates:

        compatible = []

        for reading in ocr_history:

            similarity = fuzzy_similarity(
                candidate,
                reading["text"]
            )

            if similarity >= FUZZY_THRESHOLD:

                compatible.append(
                    (
                        reading,
                        similarity
                    )
                )

        if not compatible:
            continue

        weighted_support = 0.0

        confidence_sum = 0.0

        similarity_sum = 0.0

        exact_votes = 0

        for reading, similarity in (
            compatible
        ):

            confidence_weight = max(
                0.25,
                reading["confidence"] / 100.0
            )

            weighted_support += (
                confidence_weight
                * similarity
            )

            confidence_sum += (
                reading["confidence"]
            )

            similarity_sum += similarity

            if (
                reading["text"]
                ==
                candidate
            ):

                exact_votes += 1

        count = len(
            compatible
        )

        average_confidence = (
            confidence_sum / count
        )

        average_similarity = (
            similarity_sum / count
        )

        format_score = (
            plate_format_score(
                candidate
            )
        )

        completeness_bonus = (
            format_score * 0.8
        )

        length_bonus = min(
            len(candidate) / 20.0,
            0.5
        )

        total_score = (
            weighted_support
            +
            completeness_bonus
            +
            length_bonus
            +
            exact_votes * 0.25
        )

        clusters.append(
            {
                "candidate": candidate,
                "compatible_count": count,
                "exact_votes": exact_votes,
                "weighted_support": round(
                    weighted_support,
                    3
                ),
                "average_confidence": round(
                    average_confidence,
                    2
                ),
                "average_similarity": round(
                    average_similarity,
                    3
                ),
                "format_score": round(
                    format_score,
                    3
                ),
                "total_score": round(
                    total_score,
                    3
                )
            }
        )

    if not clusters:

        return (
            None,
            0.0,
            0,
            0.0,
            []
        )

    clusters.sort(
        key=lambda x: (
            x["compatible_count"],
            x["total_score"],
            len(x["candidate"])
        ),
        reverse=True
    )

    best = clusters[0]

    candidate = best[
        "candidate"
    ]

    votes = best[
        "compatible_count"
    ]

    confidence = best[
        "average_confidence"
    ]

    similarity = (
        best[
            "average_similarity"
        ]
        * 100.0
    )

    if votes >= MIN_STABLE_READINGS:

        return (
            candidate,
            confidence,
            votes,
            similarity,
            clusters
        )

    return (
        None,
        0.0,
        votes,
        similarity,
        clusters
    )


# ================================================================
# RESET OCR
# ================================================================

def reset_ocr_history():

    global stable_plate
    global stable_confidence
    global stable_votes
    global stable_similarity

    global last_plate_text
    global last_ocr_confidence
    global last_ocr_engine

    ocr_history.clear()

    stable_plate = None
    stable_confidence = 0.0
    stable_votes = 0
    stable_similarity = 0.0

    last_plate_text = None
    last_ocr_confidence = 0.0
    last_ocr_engine = None

    print()
    print("OCR history RESET.")
    print()


# ================================================================
# SAVE JSON
# ================================================================

def save_json_files():

    # ------------------------------------------------
    # OCR history
    # ------------------------------------------------

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
            list(ocr_history),
            file,
            indent=4
        )

    # ------------------------------------------------
    # Stability clusters
    # ------------------------------------------------

    _, _, _, _, clusters = (
        stabilize_plate()
    )

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

    # ------------------------------------------------
    # Final result
    # ------------------------------------------------

    final_result = {

        "phase": "6.3",

        "system": (
            "AEGISVISION ANPR"
        ),

        "ocr_method": (
            "PaddleOCR + "
            "Tesseract fallback"
        ),

        "camera": {
            "requested_width": FRAME_WIDTH,
            "requested_height": FRAME_HEIGHT,
            "requested_fps": TARGET_FPS,
            "reported_fps": round(
                display_camera_fps,
                2
            ),
            "processing_fps": round(
                processing_fps,
                2
            )
        },

        "paddleocr_enabled_at_end": (
            PADDLE_ENABLED
        ),

        "total_frames": total_frames,

        "total_yolo_detections": (
            total_detections
        ),

        "ocr_attempts": ocr_attempts,

        "ocr_errors": ocr_errors,

        "ocr_runtime_errors": (
            ocr_runtime_errors
        ),

        "paddleocr_successes": (
            paddle_successes
        ),

        "tesseract_successes": (
            tesseract_successes
        ),

        "ocr_readings_collected": (
            len(ocr_history)
        ),

        "last_ocr_plate": (
            last_plate_text
        ),

        "last_ocr_confidence": round(
            last_ocr_confidence,
            2
        ),

        "last_ocr_engine": (
            last_ocr_engine
        ),

        "stable_plate": stable_plate,

        "stable_confidence": round(
            stable_confidence,
            2
        ),

        "stable_votes": stable_votes,

        "fuzzy_similarity": round(
            stable_similarity,
            2
        )
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

    return (
        history_path,
        clusters_path,
        final_path
    )


# ================================================================
# MAIN
# ================================================================

def main():

    global paddle_ocr
    global PADDLE_ENABLED

    global tesseract_available

    global total_frames
    global total_detections
    global ocr_attempts
    global ocr_errors

    global last_ocr_frame

    global last_plate_text
    global last_ocr_confidence
    global last_ocr_engine

    global stable_plate
    global stable_confidence
    global stable_votes
    global stable_similarity

    global fps_timer
    global fps_frame_count
    global processing_fps
    global display_camera_fps

    # ============================================================
    # HEADER
    # ============================================================

    print()
    print("=" * 70)
    print("AEGISVISION - ANPR")
    print("PHASE 6.3 - ROBUST OCR RESULT STABILIZATION")
    print("YOLO + PADDLEOCR + TESSERACT FALLBACK")
    print("FUZZY MULTI-FRAME VOTING")
    print("60 FPS WEBCAM MODE")
    print("=" * 70)
    print()

    # ============================================================
    # YOLO
    # ============================================================

    print(
        "Loading YOLO plate detector..."
    )

    try:

        detector = YOLO(
            MODEL_PATH
        )

        print(
            "YOLO detector loaded successfully."
        )

    except Exception as e:

        print()
        print(
            "ERROR: Could not load "
            "YOLO detector."
        )

        print(
            "Model:",
            MODEL_PATH
        )

        print(
            "Reason:",
            e
        )

        return

    print()

    # ============================================================
    # PADDLEOCR
    # ============================================================

    print(
        "Loading PaddleOCR..."
    )

    if PADDLE_IMPORT_AVAILABLE:

        try:

            paddle_ocr = PaddleOCR(

                lang="en",

                device="cpu",

                # Disable oneDNN / MKL-DNN
                enable_mkldnn=False,

                cpu_threads=4,

                use_doc_orientation_classify=False,

                use_doc_unwarping=False,

                use_textline_orientation=False
            )

            PADDLE_ENABLED = True

            print(
                "PaddleOCR loaded successfully."
            )

            print(
                "PaddleOCR CPU mode enabled."
            )

            print(
                "PaddleOCR MKL-DNN explicitly disabled."
            )

        except Exception as e:

            print(
                "PaddleOCR initialization failed."
            )

            print(
                "Reason:",
                e
            )

            PADDLE_ENABLED = False

    else:

        print(
            "PaddleOCR unavailable."
        )

        PADDLE_ENABLED = False

    print()

    # ============================================================
    # TESSERACT
    # ============================================================

    print(
        "Checking Tesseract OCR..."
    )

    if TESSERACT_IMPORT_AVAILABLE:

        possible_paths = [

            r"C:\Program Files\Tesseract-OCR\tesseract.exe",

            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
        ]

        tesseract_path = None

        for path in possible_paths:

            if os.path.exists(path):

                tesseract_path = path

                break

        if tesseract_path:

            try:

                pytesseract.pytesseract.tesseract_cmd = (
                    tesseract_path
                )

                version = (
                    pytesseract.get_tesseract_version()
                )

                print(
                    "Tesseract executable found:"
                )

                print(
                    tesseract_path
                )

                print(
                    "Tesseract loaded successfully."
                )

                print(
                    "Version:",
                    version
                )

                tesseract_available = True

            except Exception as e:

                print(
                    "Tesseract initialization failed."
                )

                print(
                    "Reason:",
                    e
                )

                tesseract_available = False

        else:

            print(
                "Tesseract executable not found."
            )

            tesseract_available = False

    else:

        print(
            "pytesseract is not installed."
        )

        print(
            "Install with:"
        )

        print(
            "pip install pytesseract"
        )

        tesseract_available = False

    print()

    # ============================================================
    # OPEN WEBCAM
    # ============================================================

    print(
        "Opening webcam..."
    )

    # ------------------------------------------------------------
    # DirectShow + MJPG
    # ------------------------------------------------------------

    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    if not camera.isOpened():

        print(
            "DirectShow camera opening failed."
        )

        print(
            "Trying default camera backend..."
        )

        camera = cv2.VideoCapture(
            CAMERA_INDEX
        )

    if not camera.isOpened():

        print()
        print(
            "ERROR: Could not open webcam."
        )

        return

    # ============================================================
    # REQUEST 60 FPS
    # ============================================================

    # MJPG is important for many USB webcams
    # when requesting high FPS.
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

    # ------------------------------------------------------------
    # Read actual camera settings
    # ------------------------------------------------------------

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

    actual_camera_fps = camera.get(
        cv2.CAP_PROP_FPS
    )

    display_camera_fps = (
        actual_camera_fps
    )

    print()
    print(
        "Camera opened successfully."
    )

    print("-" * 70)

    print(
        f"Resolution : "
        f"{actual_width} x "
        f"{actual_height}"
    )

    print(
        f"Requested FPS : "
        f"{TARGET_FPS}"
    )

    print(
        f"Camera FPS    : "
        f"{actual_camera_fps:.2f}"
    )

    print(
        "Video codec   : MJPG"
    )

    print("-" * 70)

    # ============================================================
    # FPS TIMER
    # ============================================================

    fps_timer = time.time()

    fps_frame_count = 0

    # ============================================================
    # START
    # ============================================================

    print()
    print(
        "REAL-TIME STABILIZED ANPR STARTED"
    )

    print("-" * 70)

    print(
        "Q / ESC = Exit"
    )

    print(
        "S       = Save current frame"
    )

    print(
        "R       = Reset OCR history"
    )

    print("-" * 70)

    print()

    # ============================================================
    # CAMERA LOOP
    # ============================================================

    while True:

        loop_start = time.time()

        ret, frame = camera.read()

        if not ret:

            print(
                "ERROR: Could not read frame."
            )

            break

        total_frames += 1

        fps_frame_count += 1

        # ========================================================
        # CALCULATE PROCESSING FPS
        # ========================================================

        current_time = time.time()

        elapsed = (
            current_time
            -
            fps_timer
        )

        if elapsed >= 1.0:

            processing_fps = (
                fps_frame_count
                /
                elapsed
            )

            fps_frame_count = 0

            fps_timer = current_time

        # ========================================================
        # YOLO
        # ========================================================

        try:

            results = detector.predict(
                source=frame,
                conf=DETECTOR_CONFIDENCE,
                verbose=False
            )

        except Exception as e:

            print(
                "YOLO detection error:",
                e
            )

            continue

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                try:

                    confidence = float(
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
                    min(
                        x1,
                        frame.shape[1] - 1
                    )
                )

                y1 = max(
                    0,
                    min(
                        y1,
                        frame.shape[0] - 1
                    )
                )

                x2 = max(
                    0,
                    min(
                        x2,
                        frame.shape[1] - 1
                    )
                )

                y2 = max(
                    0,
                    min(
                        y2,
                        frame.shape[0] - 1
                    )
                )

                width = x2 - x1

                height = y2 - y1

                if width < MIN_PLATE_WIDTH:
                    continue

                if height < MIN_PLATE_HEIGHT:
                    continue

                total_detections += 1

                detections.append(
                    {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "confidence": confidence,
                        "area": width * height
                    }
                )

        # ========================================================
        # DRAW DETECTIONS
        # ========================================================

        for detection in detections:

            x1 = detection["x1"]
            y1 = detection["y1"]
            x2 = detection["x2"]
            y2 = detection["y2"]

            confidence = detection[
                "confidence"
            ]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Plate "
                f"{confidence * 100:.1f}%",
                (
                    x1,
                    max(
                        20,
                        y1 - 8
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

        # ========================================================
        # BEST PLATE
        # ========================================================

        best_detection = None

        if detections:

            best_detection = max(
                detections,
                key=lambda d: (
                    d["confidence"],
                    d["area"]
                )
            )

        # ========================================================
        # OCR
        # ========================================================

        if (
            best_detection is not None
            and
            (
                total_frames
                -
                last_ocr_frame
            )
            >= OCR_INTERVAL
        ):

            last_ocr_frame = total_frames

            ocr_attempts += 1

            x1 = best_detection["x1"]
            y1 = best_detection["y1"]
            x2 = best_detection["x2"]
            y2 = best_detection["y2"]

            plate_crop = frame[
                y1:y2,
                x1:x2
            ]

            if (
                plate_crop is not None
                and
                plate_crop.size > 0
            ):

                ocr_results = (
                    run_combined_ocr(
                        plate_crop
                    )
                )

                # ------------------------------------------------
                # OCR SUCCESS
                # ------------------------------------------------

                if ocr_results:

                    text, confidence, engine = (
                        ocr_results[0]
                    )

                    last_plate_text = text

                    last_ocr_confidence = (
                        confidence
                    )

                    last_ocr_engine = (
                        engine
                    )

                    add_ocr_reading(
                        text,
                        confidence,
                        engine,
                        total_frames
                    )

                    # ------------------------------------------------
                    # Save crop
                    # ------------------------------------------------

                    safe_text = re.sub(
                        r"[^A-Z0-9]",
                        "_",
                        text
                    )

                    crop_filename = (
                        f"frame_"
                        f"{total_frames:06d}_"
                        f"{safe_text}.jpg"
                    )

                    crop_path = os.path.join(
                        CROP_DIR,
                        crop_filename
                    )

                    cv2.imwrite(
                        crop_path,
                        plate_crop
                    )

                    print(
                        f"[Frame {total_frames}] "
                        f"Plate: {text} | "
                        f"Detector: "
                        f"{best_detection['confidence'] * 100:.1f}% | "
                        f"OCR: "
                        f"{confidence:.1f}% | "
                        f"Engine: {engine}"
                    )

                # ------------------------------------------------
                # OCR FAILED
                # ------------------------------------------------

                else:

                    ocr_errors += 1

                    print(
                        f"[Frame {total_frames}] "
                        f"OCR attempt: "
                        f"no text detected"
                    )

        # ========================================================
        # STABILIZATION
        # ========================================================

        (
            candidate,
            candidate_confidence,
            candidate_votes,
            candidate_similarity,
            clusters
        ) = stabilize_plate()

        if candidate is not None:

            stable_plate = candidate

            stable_confidence = (
                candidate_confidence
            )

            stable_votes = (
                candidate_votes
            )

            stable_similarity = (
                candidate_similarity
            )

        # ========================================================
        # TOP INFORMATION
        # ========================================================

        overlay_y = 25

        cv2.putText(
            frame,
            "AEGISVISION ANPR - PHASE 6.3",
            (10, overlay_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )

        overlay_y += 28

        # --------------------------------------------------------
        # CAMERA FPS
        # --------------------------------------------------------

        cv2.putText(
            frame,
            f"Camera FPS: "
            f"{display_camera_fps:.1f}",
            (10, overlay_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        overlay_y += 25

        # --------------------------------------------------------
        # PROCESSING FPS
        # --------------------------------------------------------

        cv2.putText(
            frame,
            f"Processing FPS: "
            f"{processing_fps:.1f}",
            (10, overlay_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        overlay_y += 28

        # --------------------------------------------------------
        # LAST OCR
        # --------------------------------------------------------

        if last_plate_text:

            cv2.putText(
                frame,
                f"OCR: "
                f"{last_plate_text}",
                (10, overlay_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

            overlay_y += 27

            cv2.putText(
                frame,
                f"{last_ocr_engine} "
                f"{last_ocr_confidence:.1f}%",
                (10, overlay_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                2
            )

            overlay_y += 28

        # --------------------------------------------------------
        # STABLE RESULT
        # --------------------------------------------------------

        if stable_plate:

            cv2.putText(
                frame,
                f"STABLE: "
                f"{stable_plate}",
                (10, overlay_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (0, 255, 0),
                2
            )

            overlay_y += 27

            cv2.putText(
                frame,
                f"Votes: "
                f"{stable_votes}   "
                f"Confidence: "
                f"{stable_confidence:.1f}%",
                (10, overlay_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                f"STABLE: Learning "
                f"({candidate_votes}/"
                f"{MIN_STABLE_READINGS})",
                (10, overlay_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 200, 255),
                2
            )

        # ========================================================
        # BOTTOM STATUS
        # ========================================================

        engine_status = (
            "PaddleOCR"
            if PADDLE_ENABLED
            else "Tesseract Fallback"
        )

        bottom_y = (
            frame.shape[0] - 45
        )

        cv2.putText(
            frame,
            f"OCR: {engine_status}",
            (
                10,
                bottom_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Q=Exit  S=Save  R=Reset",
            (
                10,
                frame.shape[0] - 18
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        # ========================================================
        # SHOW
        # ========================================================

        cv2.imshow(
            "AEGISVISION ANPR - Phase 6.3 - 60 FPS",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        # ========================================================
        # EXIT
        # ========================================================

        if (
            key == ord("q")
            or
            key == 27
        ):

            break

        # ========================================================
        # SAVE
        # ========================================================

        elif key == ord("s"):

            timestamp = time.strftime(
                "%Y%m%d_%H%M%S"
            )

            save_path = os.path.join(
                OUTPUT_DIR,
                f"anpr_frame_{timestamp}.jpg"
            )

            cv2.imwrite(
                save_path,
                frame
            )

            print()
            print(
                "Frame saved:"
            )

            print(
                save_path
            )

            print()

        # ========================================================
        # RESET
        # ========================================================

        elif key == ord("r"):

            reset_ocr_history()

    # ============================================================
    # CLEANUP
    # ============================================================

    camera.release()

    cv2.destroyAllWindows()

    # ============================================================
    # SAVE RESULTS
    # ============================================================

    (
        history_path,
        clusters_path,
        final_path
    ) = save_json_files()

    # ============================================================
    # FINAL REPORT
    # ============================================================

    print()
    print("=" * 70)
    print("PHASE 6.3 COMPLETED")
    print("=" * 70)
    print()

    print(
        f"Total frames processed : "
        f"{total_frames}"
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
        f"OCR errors             : "
        f"{ocr_errors}"
    )

    print(
        f"OCR runtime errors     : "
        f"{ocr_runtime_errors}"
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
        f"{display_camera_fps:.2f}"
    )

    print(
        f"Processing FPS         : "
        f"{processing_fps:.2f}"
    )

    print(
        f"Last OCR plate         : "
        f"{last_plate_text}"
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

    if stable_plate:

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
            f"{stable_similarity:.2f}%"
        )

    else:

        print(
            "Stable plate           : "
            "Not confirmed"
        )

        print(
            f"Stable votes           : "
            f"{stable_votes}"
        )

        print(
            f"Fuzzy similarity       : "
            f"{stable_similarity:.2f}%"
        )

    print(
        f"OCR readings collected : "
        f"{len(ocr_history)}"
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
            history_path
        )
    )

    print()

    print(
        "Stability clusters:"
    )

    print(
        os.path.abspath(
            clusters_path
        )
    )

    print()

    print(
        "Final ANPR result:"
    )

    print(
        os.path.abspath(
            final_path
        )
    )

    print()
    print("=" * 70)
    print(
        "OCR RESULT STABILIZATION FINISHED."
    )
    print("=" * 70)


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":

    main()