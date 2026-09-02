import cv2
import json
import os
import re
import time
import threading
import queue
from collections import Counter, deque

import numpy as np
import pytesseract
from pytesseract import Output
from ultralytics import YOLO


# ============================================================
# AEGISVISION - ANPR
# PHASE 6.6 - LIVE MULTI-FRAME OCR IMPROVEMENT
# ============================================================

print("=" * 70)
print("AEGISVISION - ANPR")
print("PHASE 6.6 - LIVE MULTI-FRAME OCR IMPROVEMENT")
print("=" * 70)


# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = "models/plate_detector.pt"
OUTPUT_DIR = "output/phase6_6/multiframe_v3"
os.makedirs(OUTPUT_DIR, exist_ok=True)
FINAL_JSON = os.path.join(OUTPUT_DIR, "multiframe_anpr_result.json")
FINAL_CROP = os.path.join(OUTPUT_DIR, "best_plate_crop.jpg")

# Camera
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 60

# Runtime behavior
DEBUG = os.getenv("AEGIS_ANPR_DEBUG", "0") == "1"
DEBUG_CROP = os.getenv("AEGIS_ANPR_DEBUG_CROP", "0") == "1"
DEBUG_FRAME_INTERVAL = int(os.getenv("AEGIS_ANPR_DEBUG_FRAME_INTERVAL", "100"))

# Debug crop saving
DEBUG_CROPS_DIR = os.path.join(OUTPUT_DIR, "debug_crops")
DEBUG_CROP_LIMIT = 15
debug_crop_count = 0

# Detection / OCR cadence
YOLO_INTERVAL = 12
OCR_INTERVAL = 20
YOLO_IMAGE_SIZE = int(os.getenv("AEGIS_ANPR_YOLO_SIZE", "256"))
PAD_X_RATIO = 0.08
PAD_Y_RATIO = 0.10

# Crop / detection thresholds
MIN_CROP_WIDTH = 25
MIN_CROP_HEIGHT = 12
MIN_DETECTION_CONFIDENCE = 0.20
MIN_PLATE_WIDTH = 30
MIN_PLATE_HEIGHT = 10
MIN_PLATE_ASPECT_RATIO = 2.0
MAX_PLATE_ASPECT_RATIO = 8.0

# OCR / validation
MIN_TEXT_LENGTH = 4
MAX_TEXT_LENGTH = 10
MIN_OCR_CONFIDENCE = 35.0
MIN_VALID_FRAMES = 10
MIN_STABLE_VOTES = 3
MIN_STABILITY = 50.0

# Temporal consensus / tracking
MAX_HISTORY = 60
FUZZY_THRESHOLD = 0.78
SIMILARITY_MATCH_DISTANCE = 2
TRACK_TIMEOUT_FRAMES = 20
MAX_DETECTION_AGE = 24
IOU_THRESHOLD = 0.15

VALID_PLATE_PATTERNS = [
    r"^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$",
    r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$",
    r"^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{1,4}$",
]

# Tesseract
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ============================================================
# TESSERACT CHECK
# ============================================================
print("\nTesseract found:")
print(TESSERACT_PATH)
if not os.path.exists(TESSERACT_PATH):
    raise SystemExit(1)


# ============================================================
# LOAD YOLO
# ============================================================
print("\nLoading YOLO plate detector...")
try:
    model = YOLO(MODEL_PATH)
    print("YOLO detector loaded successfully.")
except Exception as e:
    print("ERROR loading YOLO:")
    print(e)
    raise SystemExit(1)


# ============================================================
# OCR THREAD VARIABLES
# ============================================================
ocr_queue = queue.Queue(maxsize=1)
ocr_results = queue.Queue()
ocr_thread_running = True
ocr_scheduled = 0
ocr_started = 0
ocr_completed = 0
ocr_attempts = 0
ocr_successes = 0
ocr_errors = 0
ocr_empty_results = 0
ocr_rejected_confidence = 0
ocr_rejected_validation = 0
ocr_accepted = 0
ocr_diagnostics = []
total_yolo_seconds = 0.0
total_crop_seconds = 0.0
total_preprocess_seconds = 0.0
total_ocr_seconds = 0.0
total_frame_process_seconds = 0.0

history = deque(maxlen=MAX_HISTORY)
latest_raw_text = ""
latest_normalized_text = ""
last_ocr_confidence = 0.0
last_detection_confidence = 0.0
best_crop = None
best_crop_quality = 0.0
track_history = {}
track_id = 0
best_track_id = 0
debug_crop_count = 0


# ============================================================
# TEXT CLEANING
# ============================================================
def clean_text(text):
    if text is None:
        return ""
    text = str(text).upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def normalize_plate(text):
    text = clean_text(text)
    if not text:
        return ""
    if len(text) < MIN_TEXT_LENGTH or len(text) > MAX_TEXT_LENGTH:
        return ""
    return text


LETTER_FIXES = {"0": "O", "1": "I", "5": "S", "6": "G", "8": "B", "2": "Z"}
DIGIT_FIXES = {"O": "0", "I": "1", "L": "1", "S": "5", "B": "8", "G": "6", "Z": "2"}
KNOWN_STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "GA", "GJ", "HP", "HR",
    "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL",
    "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB",
}


def normalize_registration_candidate(text):
    """Apply glyph corrections only at positions required by a plate structure."""
    raw = normalize_plate(text)
    if not raw:
        return ""

    best = None
    for district_length in (1, 2):
        for series_length in (1, 2, 3):
            number_length = len(raw) - 2 - district_length - series_length
            if not 1 <= number_length <= 4:
                continue
            state = raw[:2]
            district = raw[2:2 + district_length]
            series_start = 2 + district_length
            series = raw[series_start:series_start + series_length]
            number = raw[series_start + series_length:]
            corrected = []
            replacements = 0

            for char in state:
                if char.isalpha():
                    corrected.append(char)
                elif char in LETTER_FIXES:
                    corrected.append(LETTER_FIXES[char])
                    replacements += 1
                else:
                    corrected = []
                    break
            if len(corrected) != 2:
                continue

            for char in district:
                if char.isdigit():
                    corrected.append(char)
                elif char in DIGIT_FIXES:
                    corrected.append(DIGIT_FIXES[char])
                    replacements += 1
                else:
                    corrected = []
                    break
            if len(corrected) != 2 + district_length:
                continue

            for char in series:
                if char.isalpha():
                    corrected.append(char)
                elif char in LETTER_FIXES:
                    corrected.append(LETTER_FIXES[char])
                    replacements += 1
                else:
                    corrected = []
                    break
            if len(corrected) != 2 + district_length + series_length:
                continue

            for char in number:
                if char.isdigit():
                    corrected.append(char)
                elif char in DIGIT_FIXES:
                    corrected.append(DIGIT_FIXES[char])
                    replacements += 1
                else:
                    corrected = []
                    break
            if len(corrected) != len(raw):
                continue

            candidate = "".join(corrected)
            score = plate_format_score(candidate) - (replacements * 0.03)
            if best is None or score > best[0]:
                best = (score, candidate)

    return best[1] if best else raw


def plate_format_score(text):
    plate = clean_text(text)
    if not plate or len(plate) < 6 or len(plate) > 10:
        return 0.0
    if plate[:2] not in KNOWN_STATE_CODES:
        return 0.0
    state_score = 1.0
    match = re.fullmatch(r"([A-Z]{2})([0-9]{1,2})([A-Z]{1,3})([0-9]{1,4})", plate)
    if not match:
        return 0.0
    _, district, series, number = match.groups()
    structure_score = 0.35
    structure_score += 0.20 if len(district) == 2 else 0.12
    structure_score += 0.20 if 1 <= len(series) <= 2 else 0.12
    structure_score += 0.25 if len(number) == 4 else 0.15
    return min(1.0, 0.55 * state_score + 0.45 * structure_score)


def validate_indian_plate(text):
    return plate_format_score(normalize_registration_candidate(text)) >= 0.75


def levenshtein_distance(a, b):
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (ca != cb)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def similarity(a, b):
    if not a or not b:
        return 0.0
    distance = levenshtein_distance(a, b)
    max_length = max(len(a), len(b))
    if max_length == 0:
        return 1.0
    return 1.0 - (distance / max_length)


OCR_CONFUSION_PAIRS = {
    frozenset(("O", "0")),
    frozenset(("I", "1")),
    frozenset(("L", "I")),
    frozenset(("L", "1")),
    frozenset(("S", "5")),
    frozenset(("B", "8")),
    frozenset(("G", "6")),
    frozenset(("Z", "2")),
}


def confusion_aware_distance(a, b):
    a = clean_text(a)
    b = clean_text(b)
    if not a or not b:
        return max(len(a), len(b))

    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            if char_a == char_b:
                substitution_cost = 0.0
            elif frozenset((char_a, char_b)) in OCR_CONFUSION_PAIRS:
                substitution_cost = 0.15
            else:
                substitution_cost = 1.0
            current.append(min(
                current[j - 1] + 1.0,
                previous[j] + 1.0,
                previous[j - 1] + substitution_cost,
            ))
        previous = current
    return previous[-1]


def is_similar_plate(a, b):
    if not a or not b:
        return False
    a = clean_text(a)
    b = clean_text(b)
    if not a or not b:
        return False
    if abs(len(a) - len(b)) > 1:
        return False
    distance = confusion_aware_distance(a, b)
    score = 1.0 - (distance / max(len(a), len(b)))
    if score >= FUZZY_THRESHOLD:
        return True
    return distance <= SIMILARITY_MATCH_DISTANCE and min(len(a), len(b)) >= 6


# ============================================================
# CROP QUALITY / PREPROCESSING
# ============================================================
def calculate_crop_quality(crop, detection_confidence=0.0):
    if crop is None or crop.size == 0:
        return 0.0
    h, w = crop.shape[:2]
    if w <= 0 or h <= 0:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = gray.mean() / 255.0
    contrast = gray.std() / 255.0

    size_score = min((w * h) / 5000.0, 1.0)
    sharpness_score = min(sharpness / 500.0, 1.0)
    contrast_score = min(contrast, 1.0)
    brightness_score = 1.0 - abs(brightness - 0.55) / 0.55
    brightness_score = max(0.0, min(1.0, brightness_score))

    aspect_ratio = w / max(h, 1)
    aspect_score = 1.0 - min(abs(aspect_ratio - 4.0) / 4.0, 1.0)
    conf_score = min(float(detection_confidence), 1.0)

    quality = (
        0.30 * size_score +
        0.25 * sharpness_score +
        0.15 * contrast_score +
        0.15 * brightness_score +
        0.10 * aspect_score +
        0.05 * conf_score
    )
    return float(np.clip(quality, 0.0, 1.0))


def is_plate_candidate_crop(crop, detection_confidence=0.0):
    if crop is None or crop.size == 0:
        return False
    h, w = crop.shape[:2]
    if w < MIN_PLATE_WIDTH or h < MIN_PLATE_HEIGHT:
        return False
    aspect_ratio = w / max(h, 1)
    if aspect_ratio < MIN_PLATE_ASPECT_RATIO or aspect_ratio > MAX_PLATE_ASPECT_RATIO:
        return False
    if detection_confidence < MIN_DETECTION_CONFIDENCE:
        return False
    return True


def create_preprocessed_images(crop):
    if crop is None or crop.size == 0:
        return []

    h, w = crop.shape[:2]
    scale = max(2.0, 300.0 / max(w, 1), 100.0 / max(h, 1))
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    return [("enhanced", enhanced)]


# ============================================================
# OCR
# ============================================================
def run_tesseract(image):
    readings = []
    try:
        config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        raw = pytesseract.image_to_string(image, config=config)
        text = clean_text(raw)
        if not text:
            return readings

        normalized = normalize_registration_candidate(text)
        if not normalized:
            return readings

        validation = validate_indian_plate(normalized)
        confidence = 80.0 if validation else 40.0

        diagnostic = {
            "preprocessing": "enhanced",
            "psm": 7,
            "raw_text": raw,
            "ocr_confidence": round(float(confidence), 2),
            "cleaned_text": text,
            "validation": validation,
            "reason": "",
        }

        if confidence < MIN_OCR_CONFIDENCE and len(normalized) < 6:
            diagnostic["reason"] = "rejected_confidence"
            readings.append({"diagnostic": diagnostic})
            return readings

        if not validation:
            diagnostic["reason"] = "rejected_validation"
            readings.append({"diagnostic": diagnostic})
            return readings

        diagnostic["reason"] = "accepted_candidate"
        readings.append({
            "raw": text,
            "normalized": normalized,
            "preprocessing": "enhanced",
            "psm": 7,
            "confidence": confidence,
            "diagnostic": diagnostic,
        })
    except Exception as error:
        if DEBUG:
            print(f"[TESSERACT ERROR] {error}")
    return readings


def save_debug_crops(request, original_crop, padded_crop, final_image, raw_text):
    global debug_crop_count
    if not DEBUG_CROP:
        return
    if debug_crop_count >= DEBUG_CROP_LIMIT:
        return
    debug_crop_count += 1
    frame_num = request.get("frame", 0)
    base = os.path.join(DEBUG_CROPS_DIR, f"frame_{frame_num:06d}")
    os.makedirs(base, exist_ok=True)
    if original_crop is not None and original_crop.size > 0:
        cv2.imwrite(os.path.join(base, "01_original_yolo_crop.jpg"), original_crop)
    if padded_crop is not None and padded_crop.size > 0:
        cv2.imwrite(os.path.join(base, "02_padded_crop.jpg"), padded_crop)
    if final_image is not None and final_image.size > 0:
        if len(final_image.shape) == 2:
            cv2.imwrite(os.path.join(base, "03_final_ocr_image.jpg"), final_image)
        else:
            cv2.imwrite(os.path.join(base, "03_final_ocr_image.jpg"), final_image)
    with open(os.path.join(base, "ocr.txt"), "w", encoding="utf-8") as f:
        f.write(f"raw_text: {raw_text}\n")


def ocr_worker():
    global ocr_started, ocr_completed, ocr_errors, total_ocr_seconds, total_preprocess_seconds
    while True:
        try:
            request = ocr_queue.get(timeout=0.2)
        except queue.Empty:
            if not ocr_thread_running:
                break
            continue

        if request is None:
            ocr_queue.task_done()
            break

        ocr_started += 1
        ocr_start = time.perf_counter()
        try:
            crop = request["crop"]
            original_crop = request.get("original_yolo_crop")
            preprocess_start = time.perf_counter()
            variants = create_preprocessed_images(crop)
            total_preprocess_seconds += time.perf_counter() - preprocess_start

            results = []
            for preprocessing_name, image in variants:
                results.extend(run_tesseract(image))

            if (DEBUG_CROP and debug_crop_count < DEBUG_CROP_LIMIT and results
                    and request.get("frame", 0) % DEBUG_FRAME_INTERVAL == 0):
                final_image = variants[0][1] if variants else crop
                best_raw = results[0].get("diagnostic", {}).get("raw_text", "") if results else ""
                save_debug_crops(request, original_crop, crop, final_image, best_raw)

            ocr_results.put({"request": request, "readings": results})
        except Exception as e:
            ocr_errors += 1
            ocr_results.put({"request": request, "error": str(e), "readings": []})
        finally:
            total_ocr_seconds += time.perf_counter() - ocr_start
            ocr_completed += 1
            ocr_queue.task_done()


ocr_thread = threading.Thread(target=ocr_worker, daemon=True)
ocr_thread.start()


# ============================================================
# FRAME SELECTION / TEMPORAL FUZZY GROUPING
# ============================================================
def candidate_score(candidate, frequency, psm_agreement, preprocessing_agreement, crop_quality, temporal_similarity):
    confidence = min(float(candidate.get("confidence", 0.0)) / 100.0, 1.0)
    format_score = plate_format_score(candidate.get("normalized", ""))
    length_score = min(len(candidate.get("normalized", "")) / 10.0, 1.0)
    frequency_score = min(frequency / 4.0, 1.0)
    psm_score = min(psm_agreement / 3.0, 1.0)
    preprocessing_score = min(preprocessing_agreement / 4.0, 1.0)
    return (
        0.30 * confidence
        + 0.18 * frequency_score
        + 0.12 * psm_score
        + 0.10 * preprocessing_score
        + 0.18 * format_score
        + 0.06 * temporal_similarity
        + 0.03 * length_score
        + 0.03 * crop_quality
    )


def select_best_frame_reading(ocr_readings, crop_quality=0.5):
    candidates = []
    for item in ocr_readings:
        raw = clean_text(item.get("raw", ""))
        normalized = normalize_registration_candidate(raw)
        if not normalized:
            continue
        existing = next((candidate for candidate in candidates if candidate["normalized"] == normalized), None)
        if existing is None:
            candidates.append({
                "raw": raw,
                "normalized": normalized,
                "confidence": float(item.get("confidence", 0.0)),
                "frequency": 1,
                "psm_modes": {item.get("psm", 0)},
                "preprocessing_modes": {item.get("preprocessing", "unknown")},
                "quality_score": crop_quality,
            })
        else:
            existing["frequency"] += 1
            existing["confidence"] = max(existing["confidence"], float(item.get("confidence", 0.0)))
            existing["psm_modes"].add(item.get("psm", 0))
            existing["preprocessing_modes"].add(item.get("preprocessing", "unknown"))

    if not candidates:
        return None

    groups = []
    for candidate in candidates:
        matching = [
            group for group in groups
            if any(is_similar_plate(candidate["normalized"], member["normalized"]) for member in group)
        ]
        if matching:
            matching[0].append(candidate)
        else:
            groups.append([candidate])

    previous_plate = history[-1] if history else ""
    scored_groups = []
    for group in groups:
        frequency = sum(item["frequency"] for item in group)
        psm_agreement = len({psm for item in group for psm in item["psm_modes"]})
        preprocessing_agreement = len({name for item in group for name in item["preprocessing_modes"]})
        group_similarity = 0.0
        if previous_plate:
            group_similarity = 1.0 - (
                confusion_aware_distance(group[0]["normalized"], previous_plate)
                / max(len(group[0]["normalized"]), len(previous_plate))
            )
        for candidate in group:
            candidate["frequency"] = frequency
            candidate["psm_agreement"] = psm_agreement
            candidate["format_score"] = plate_format_score(candidate["normalized"])
            candidate["similarity_score"] = group_similarity
            candidate["preprocessing_agreement"] = preprocessing_agreement
            candidate["score"] = candidate_score(
                candidate,
                frequency,
                psm_agreement,
                preprocessing_agreement,
                crop_quality,
                group_similarity,
            )
        scored_groups.append(group)

    scored_groups.sort(
        key=lambda group: (
            max(item["score"] for item in group),
            len(group),
            max(item["confidence"] for item in group),
        ),
        reverse=True,
    )
    winning_group = scored_groups[0]
    selected = max(
        winning_group,
        key=lambda item: (item["score"], item["confidence"], item["frequency"]),
    )
    consensus = []
    for position in range(max(len(item["normalized"]) for item in winning_group)):
        votes = Counter()
        for item in winning_group:
            if position >= len(item["normalized"]):
                continue
            weight = item["frequency"] * max(item["confidence"] / 100.0, 0.1)
            votes[item["normalized"][position]] += weight
        if votes:
            consensus.append(votes.most_common(1)[0][0])
    consensus_text = normalize_registration_candidate("".join(consensus))
    if consensus_text and plate_format_score(consensus_text) >= 0.75:
        selected["normalized"] = consensus_text
        selected["raw"] = consensus_text
    selected["all_candidates"] = [item for group in scored_groups for item in group]
    return selected


def build_temporal_clusters():
    clusters = []
    for text in history:
        if not text:
            continue
        matching_indexes = [
            index
            for index, cluster in enumerate(clusters)
            if any(is_similar_plate(text, item) for item in cluster["items"])
        ]
        if not matching_indexes:
            clusters.append({"representative": text, "items": [text]})
            continue

        primary = clusters[matching_indexes[0]]
        primary["items"].append(text)
        for index in reversed(matching_indexes[1:]):
            primary["items"].extend(clusters[index]["items"])
            del clusters[index]

    for cluster in clusters:
        cluster["representative"] = Counter(cluster["items"]).most_common(1)[0][0]
    clusters.sort(key=lambda cluster: len(cluster["items"]), reverse=True)
    return clusters


def print_temporal_state():
    clusters = build_temporal_clusters()
    print("Clusters:")
    for index, cluster in enumerate(clusters, 1):
        print(f"cluster {index} -> {cluster['items']}")
        print(f"votes -> {len(cluster['items'])}")


def calculate_stable_plate():
    clusters = build_temporal_clusters()
    if not clusters:
        return ("", 0, 0.0, 0)

    best_group = clusters[0]
    plate = Counter(best_group["items"]).most_common(1)[0][0]
    votes = len(best_group["items"])
    valid_frame_readings = len(history)
    stability = (votes / max(valid_frame_readings, 1)) * 100.0
    return (plate, votes, stability, valid_frame_readings)


def get_status_from_stability(stability, valid_frame_readings):
    if valid_frame_readings < MIN_VALID_FRAMES:
        return "NEEDS_IMPROVEMENT"
    if stability >= 90.0:
        return "STABLE"
    if stability >= 75.0:
        return "GOOD"
    if stability >= 50.0:
        return "MODERATE"
    return "NEEDS_IMPROVEMENT"


# ============================================================
# TRACKING
# ============================================================
def box_iou(box_a, box_b):
    if box_a is None or box_b is None:
        return 0.0
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def track_plate(current_plate, box, current_time):
    global track_id, best_track_id
    if not current_plate or box is None:
        return 0

    if not track_history:
        track_id += 1
        track_history[track_id] = {
            "plate": current_plate,
            "box": box,
            "last_seen": current_time,
            "history": deque([current_plate], maxlen=MAX_HISTORY),
        }
        best_track_id = track_id
        return track_id

    best_match = None
    best_score = -1.0
    for tid, info in track_history.items():
        prev_plate = info["plate"]
        prev_box = info["box"]
        if prev_plate == current_plate:
            best_match = tid
            best_score = 1.0
            break

        if is_similar_plate(prev_plate, current_plate):
            iou = box_iou(prev_box, box)
            center_prev = ((prev_box[0] + prev_box[2]) / 2.0, (prev_box[1] + prev_box[3]) / 2.0)
            center_cur = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
            center_dist = np.hypot(center_prev[0] - center_cur[0], center_prev[1] - center_cur[1])
            prev_size = max(1, prev_box[2] - prev_box[0]) * max(1, prev_box[3] - prev_box[1])
            cur_size = max(1, box[2] - box[0]) * max(1, box[3] - box[1])
            size_ratio = min(prev_size, cur_size) / max(prev_size, cur_size)
            score = (iou * 0.5) + (size_ratio * 0.3) + max(0.0, 1.0 - center_dist / 200.0) * 0.2
            if score > best_score:
                best_score = score
                best_match = tid

    if best_match is not None and best_score >= IOU_THRESHOLD:
        info = track_history[best_match]
        info.update({"plate": current_plate, "box": box, "last_seen": current_time})
        info.setdefault("history", deque(maxlen=MAX_HISTORY)).append(current_plate)
        best_track_id = best_match
        return best_match

    track_id += 1
    track_history[track_id] = {
        "plate": current_plate,
        "box": box,
        "last_seen": current_time,
        "history": deque([current_plate], maxlen=MAX_HISTORY),
    }
    best_track_id = track_id
    return track_id


def prune_tracks(current_time):
    for tid in list(track_history.keys()):
        last_seen = track_history[tid].get("last_seen", current_time)
        if current_time - last_seen > TRACK_TIMEOUT_FRAMES / 30.0:
            del track_history[tid]


# ============================================================
# LIVE LOOP
# ============================================================
print("\nOpening webcam...")
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    ocr_thread_running = False
    raise SystemExit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Resolution : {actual_width} x {actual_height}")
print(f"YOLO size  : {YOLO_IMAGE_SIZ}")
print(f"YOLO every : {YOLO_INTERVAL} frames")
print(f"OCR every  : {OCR_INTERVAL} frames")
print("-" * 70)

print("\nControls:")
print("Q / ESC = Quit")
print("S       = Save best plate crop")
print("R       = Reset OCR history")

print("\n" + "=" * 70)
print("LIVE OCR STABILITY TEST STARTED")
print("=" * 70)


# ============================================================
# MAIN LOOP
# ============================================================
total_frames = 0
yolo_runs = 0
yolo_detections = 0
start_time = time.time()
current_box = None
saved_crop = False
last_best_frame_result = None
yolo_crop_diagnostics = []
detection_age = MAX_DETECTION_AGE + 1

while True:
    frame_start = time.perf_counter()
    ret, frame = cap.read()
    if not ret:
        print("\nCamera frame read failed.")
        break

    total_frames += 1

    run_detection = total_frames == 1 or total_frames % YOLO_INTERVAL == 0
    if run_detection:
        yolo_runs += 1
        yolo_start = time.perf_counter()
        try:
            results = model.predict(source=frame, imgsz=YOLO_IMAGE_SIZE, conf=MIN_DETECTION_CONFIDENCE, verbose=False)
            boxes = results[0].boxes
            best_box = None
            best_confidence = 0.0
            if boxes is not None:
                for box in boxes:
                    confidence = float(box.conf[0])
                    if confidence > best_confidence:
                        best_box = box.xyxy[0].cpu().numpy()
                        best_confidence = confidence
            if best_box is not None:
                current_box = best_box
                last_detection_confidence = best_confidence
                detection_age = 0
                yolo_detections += 1
                bx1, by1, bx2, by2 = map(int, best_box)
                bbox_w = max(1, bx2 - bx1)
                bbox_h = max(1, by2 - by1)
                touches_boundary = (bx1 <= 0 or by1 <= 0 or bx2 >= frame.shape[1] - 1 or by2 >= frame.shape[0] - 1)
                yolo_crop_diagnostics.append({
                    "frame": total_frames,
                    "bbox_x1": int(bx1),
                    "bbox_y1": int(by1),
                    "bbox_x2": int(bx2),
                    "bbox_y2": int(by2),
                    "bbox_width": bbox_w,
                    "bbox_height": bbox_h,
                    "aspect_ratio": round(bbox_w / max(bbox_h, 1), 2),
                    "padded_width": bbox_w + int(bbox_w * PAD_X_RATIO) * 2,
                    "padded_height": bbox_h + int(bbox_h * PAD_Y_RATIO) * 2,
                    "touches_boundary": touches_boundary,
                    "detection_confidence": round(float(best_confidence), 4),
                })
            else:
                detection_age = MAX_DETECTION_AGE + 1
        except Exception:
            detection_age = MAX_DETECTION_AGE + 1
        finally:
            total_yolo_seconds += time.perf_counter() - yolo_start

    if not run_detection and current_box is not None:
        detection_age += 1
    if detection_age > MAX_DETECTION_AGE:
        current_box = None
        last_detection_confidence = 0.0

    current_crop = None
    original_yolo_crop = None
    crop_diag = {}
    if current_box is not None:
        crop_start = time.perf_counter()
        x1, y1, x2, y2 = map(int, current_box)
        box_width = max(1, x2 - x1)
        box_height = max(1, y2 - y1)
        
        original_yolo_crop = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)].copy()
        
        pad_x = max(2, int(box_width * PAD_X_RATIO))
        pad_y = max(2, int(box_height * PAD_Y_RATIO))
        x1 = max(0, min(x1 - pad_x, frame.shape[1] - 1))
        y1 = max(0, min(y1 - pad_y, frame.shape[0] - 1))
        x2 = max(0, min(x2 + pad_x, frame.shape[1]))
        y2 = max(0, min(y2 + pad_y, frame.shape[0]))
        if x2 > x1 and y2 > y1:
            current_crop = frame[y1:y2, x1:x2].copy()
            crop_diag = {
                "crop_width": current_crop.shape[1],
                "crop_height": current_crop.shape[0],
                "crop_aspect_ratio": round(current_crop.shape[1] / max(current_crop.shape[0], 1), 2),
            }
            if is_plate_candidate_crop(current_crop, last_detection_confidence):
                quality = calculate_crop_quality(current_crop, last_detection_confidence) if run_detection else 0.0
                if quality > best_crop_quality:
                    best_crop_quality = quality
                    best_crop = current_crop.copy()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Plate {last_detection_confidence * 100:.1f}%", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        total_crop_seconds += time.perf_counter() - crop_start

    # OCR request
    if current_crop is not None and total_frames % OCR_INTERVAL == 0:
        crop_h, crop_w = current_crop.shape[:2]
        crop_quality = calculate_crop_quality(current_crop, last_detection_confidence)
        if (crop_w >= MIN_CROP_WIDTH and crop_h >= MIN_CROP_HEIGHT
                and is_plate_candidate_crop(current_crop, last_detection_confidence)):
            request = {
                "frame": total_frames,
                "crop": current_crop.copy(),
                "original_yolo_crop": original_yolo_crop.copy() if original_yolo_crop is not None else None,
                "crop_width": crop_w,
                "crop_height": crop_h,
                "detection_confidence": last_detection_confidence,
                "crop_quality": crop_quality,
                "crop_diag": crop_diag,
            }
            try:
                ocr_queue.put_nowait(request)
                ocr_scheduled += 1
                ocr_attempts = ocr_scheduled
            except queue.Full:
                ocr_diagnostics.append({
                    "frame": total_frames,
                    "crop_width": crop_w,
                    "crop_height": crop_h,
                    "yolo_confidence": round(float(last_detection_confidence), 4),
                    "crop_quality": round(float(crop_quality), 4),
                    "reason": "queue_full_request_not_scheduled",
                })
        elif DEBUG:
            print(f"[OCR SKIP] frame={total_frames} crop={crop_w}x{crop_h} "
                  f"det_conf={last_detection_confidence:.3f} quality={crop_quality:.3f} "
                  "reason=invalid_crop_geometry_or_confidence")

    while True:
        try:
            result = ocr_results.get_nowait()
        except queue.Empty:
            break

        request = result.get("request", {}) if isinstance(result, dict) else {}
        readings = result.get("readings", []) if isinstance(result, dict) else []
        diagnostics = [item["diagnostic"] for item in readings if "diagnostic" in item]
        for diagnostic in diagnostics:
            diagnostic_entry = {
                "frame": request.get("frame"),
                "crop_width": request.get("crop_width"),
                "crop_height": request.get("crop_height"),
                "yolo_confidence": round(float(request.get("detection_confidence", 0.0)), 4),
                "crop_quality": round(float(request.get("crop_quality", 0.0)), 4),
                **diagnostic,
            }
            ocr_diagnostics.append(diagnostic_entry)
            reason = diagnostic.get("reason", "")
            if reason == "empty_tesseract_text":
                ocr_empty_results += 1
            elif reason == "rejected_confidence":
                ocr_rejected_confidence += 1
            elif reason == "rejected_validation":
                ocr_rejected_validation += 1

            if DEBUG:
                print(
                    f"[OCR DIAGNOSTIC] frame={diagnostic_entry['frame']} "
                    f"crop={diagnostic_entry['crop_width']}x{diagnostic_entry['crop_height']} "
                    f"det={diagnostic_entry['yolo_confidence']:.3f} "
                    f"quality={diagnostic_entry['crop_quality']:.3f} "
                    f"variant={diagnostic.get('preprocessing')} psm={diagnostic.get('psm')} "
                    f"raw={diagnostic.get('raw_text', '')!r} "
                    f"conf={diagnostic.get('ocr_confidence', 0.0):.1f} "
                    f"cleaned={diagnostic.get('cleaned_text', '')!r} "
                    f"valid={diagnostic.get('validation')} reason={reason}"
                )

        if not isinstance(result, dict) or "error" in result:
            if "error" in result:
                ocr_diagnostics.append({
                    "frame": request.get("frame"),
                    "crop_width": request.get("crop_width"),
                    "crop_height": request.get("crop_height"),
                    "yolo_confidence": round(float(request.get("detection_confidence", 0.0)), 4),
                    "crop_quality": round(float(request.get("crop_quality", 0.0)), 4),
                    "reason": f"ocr_worker_exception: {result.get('error', 'Unknown error')}",
                })
                if DEBUG:
                    print("[OCR ERROR]", result.get("error", "Unknown error"))
            continue

        if not readings:
            if DEBUG:
                print("[OCR RESULT] No text detected")
            continue

        crop_quality = float(request.get("crop_quality", 0.5))
        best_result = select_best_frame_reading(readings, crop_quality)
        if best_result is None:
            if DEBUG:
                print("[OCR RESULT] No usable OCR reading")
            continue

        normalized = normalize_registration_candidate(best_result.get("raw", ""))
        if not normalized:
            continue

        selected = {
            "raw": best_result.get("raw", normalized),
            "normalized": normalized,
            "confidence": float(best_result.get("confidence", 0.0)),
            "quality": float(best_result.get("quality_score", crop_quality)),
            "score": float(best_result.get("score", 0.0)),
            "preprocessing": best_result.get("preprocessing", "unknown"),
            "psm": best_result.get("psm", 7),
            "frame": request.get("frame", total_frames),
        }

        if DEBUG:
            print("\n--------------------------------------------------")
            print("[OCR CANDIDATES]")
            print("candidate | confidence | frequency | psm_agreement | preprocessing_agreement | format_score | similarity_score | final_score")
            for item in best_result.get("all_candidates", [best_result]):
                print(
                    f"{item.get('normalized', '')} | "
                    f"{item.get('confidence', 0.0):.1f} | "
                    f"{item.get('frequency', 0)} | "
                    f"{len(item.get('psm_modes', []))} | "
                    f"{len(item.get('preprocessing_modes', []))} | "
                    f"{item.get('format_score', 0.0):.2f} | "
                    f"{item.get('similarity_score', 0.0):.2f} | "
                    f"{item.get('score', 0.0):.3f}"
                )
            print(
                f"[OCR SELECTED] {selected['normalized']} "
                f"reason=confidence={selected['confidence']:.1f}, "
                f"frequency={best_result.get('frequency', 0)}, "
                f"psm_agreement={best_result.get('psm_agreement', 0)}, "
                f"format_score={best_result.get('format_score', 0.0):.2f}, "
                f"final_score={selected['score']:.3f}"
            )
            print("--------------------------------------------------")

        latest_raw_text = selected["raw"]
        latest_normalized_text = selected["normalized"]
        last_ocr_confidence = selected["confidence"]
        ocr_successes += 1
        ocr_accepted += 1

        if selected["normalized"] and validate_indian_plate(selected["normalized"]):
            history.append(selected["normalized"])
            print(f"OCR reading: frame {selected['frame']} -> {selected['normalized']}")
            print_temporal_state()

        stable_plate, stable_votes, stability, valid_frame_readings = calculate_stable_plate()
        status = get_status_from_stability(stability, valid_frame_readings)

        if current_box is not None:
            best_track_id = track_plate(stable_plate if stable_plate else selected["normalized"], current_box.tolist(), time.time())
        prune_tracks(time.time())

        last_best_frame_result = selected

    total_frame_process_seconds += time.perf_counter() - frame_start

    stable_plate, stable_votes, stability, valid_frame_readings = calculate_stable_plate()
    status = get_status_from_stability(stability, valid_frame_readings)
    display_plate = stable_plate if stable_plate else "UNKNOWN"
    if stable_plate and last_ocr_confidence < MIN_OCR_CONFIDENCE:
        display_plate = "UNKNOWN"

    cv2.putText(frame, f"Plate: {display_plate}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 0), 2)
    cv2.putText(frame, f"Stability: {stability:.1f}% | Valid: {valid_frame_readings}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(frame, f"OCR success rate: {((ocr_successes / max(ocr_attempts, 1)) * 100.0):.1f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, f"Status: {status}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, f"Track: {best_track_id}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, f"Det conf: {last_detection_confidence:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
    cv2.putText(frame, f"OCR conf: {last_ocr_confidence:.1f}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
    cv2.imshow("AEGISVISION ANPR - LIVE", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27):
        break
    elif key == ord("s"):
        if best_crop is not None:
            cv2.imwrite(FINAL_CROP, best_crop)
            print("\n[SAVED]", FINAL_CROP)
        elif current_crop is not None:
            cv2.imwrite(FINAL_CROP, current_crop)
            print("\n[SAVED]", FINAL_CROP)
        else:
            print("\n[SAVE] No plate crop available.")
    elif key == ord("r"):
        history.clear()
        latest_raw_text = ""
        latest_normalized_text = ""
        print("\n[RESET] OCR history cleared.")


# ============================================================
# CLEANUP / JSON OUTPUT
# ============================================================
cap.release()
cv2.destroyAllWindows()
ocr_thread_running = False
ocr_queue.put(None)
ocr_queue.join()
ocr_thread.join(timeout=2)

# The worker may finish after the last camera frame, so consume all results
# before calculating the final frame-level metrics.
while not ocr_results.empty():
    result = ocr_results.get_nowait()
    if isinstance(result, dict) and "error" not in result:
        request = result.get("request", {})
        readings = result.get("readings", [])
        if readings:
            selected = select_best_frame_reading(
                readings,
                float(request.get("crop_quality", 0.5)),
            )
            if selected and validate_indian_plate(selected.get("normalized", "")):
                history.append(selected["normalized"])
                ocr_successes += 1
                ocr_accepted += 1
                print(f"OCR reading: frame {request.get('frame')} -> {selected['normalized']}")
                print_temporal_state()

elapsed_time = time.time() - start_time
processing_fps = total_frames / elapsed_time if elapsed_time > 0 else 0.0
stable_plate, stable_votes, stability, valid_frame_readings = calculate_stable_plate()
status = get_status_from_stability(stability, valid_frame_readings)
reported_plate = stable_plate
final_reason = ""
if not reported_plate:
    reported_plate = "UNKNOWN"
    final_reason = "no validated temporal cluster"
elif last_ocr_confidence < MIN_OCR_CONFIDENCE:
    reported_plate = "UNKNOWN"
    final_reason = "insufficient OCR confidence"
elif stable_votes < MIN_STABLE_VOTES:
    reported_plate = "UNKNOWN"
    final_reason = "insufficient temporal votes"

result_data = {
    "total_frames": total_frames,
    "yolo_runs": yolo_runs,
    "yolo_detections": yolo_detections,
    "ocr_attempts": ocr_attempts,
    "ocr_scheduled": ocr_scheduled,
    "ocr_started": ocr_started,
    "ocr_completed": ocr_completed,
    "ocr_successes": ocr_successes,
    "ocr_errors": ocr_errors,
    "ocr_empty_results": ocr_empty_results,
    "ocr_rejected_confidence": ocr_rejected_confidence,
    "ocr_rejected_validation": ocr_rejected_validation,
    "ocr_accepted": ocr_accepted,
    "ocr_diagnostics": ocr_diagnostics,
    "ocr_readings": len(history),
    "yolo_crop_diagnostics": yolo_crop_diagnostics[-50:],
    "final_plate": reported_plate,
    "consensus_plate": stable_plate,
    "final_plate_reason": final_reason,
    "stable_votes": stable_votes,
    "valid_frame_readings": valid_frame_readings,
    "stability_percent": round(float(stability), 2),
    "ocr_success_rate": round((ocr_successes / max(ocr_attempts, 1)) * 100.0, 2),
    "best_detection_confidence": round(float(last_detection_confidence), 4),
    "best_ocr_confidence": round(float(last_ocr_confidence), 2),
    "total_yolo_seconds": round(float(total_yolo_seconds), 3),
    "average_yolo_ms": round((total_yolo_seconds / max(yolo_runs, 1)) * 1000.0, 2),
    "total_crop_seconds": round(float(total_crop_seconds), 3),
    "total_preprocess_seconds": round(float(total_preprocess_seconds), 3),
    "average_preprocess_ms": round((total_preprocess_seconds / max(ocr_completed, 1)) * 1000.0, 2),
    "total_ocr_seconds": round(float(total_ocr_seconds), 3),
    "average_ocr_ms": round((total_ocr_seconds / max(ocr_completed, 1)) * 1000.0, 2),
    "total_frame_process_seconds": round(float(total_frame_process_seconds), 3),
    "average_frame_process_ms": round((total_frame_process_seconds / max(total_frames, 1)) * 1000.0, 2),
    "processing_fps": round(float(processing_fps), 2),
    "status": status,
    "track_id": best_track_id,
    "latest_raw_text": latest_raw_text,
    "latest_normalized_text": latest_normalized_text,
    "debug_crops_saved": debug_crop_count,
}

with open(FINAL_JSON, "w", encoding="utf-8") as f:
    json.dump(result_data, f, indent=4)

print("\n" + "=" * 70)
print("## LIVE ANPR SUMMARY")
print("Frames:", total_frames)
print("YOLO detections:", yolo_detections)
print("OCR scheduled:", ocr_scheduled)
print("OCR started:", ocr_started)
print("OCR completed:", ocr_completed)
print("OCR successes:", ocr_successes)
print("OCR exceptions:", ocr_errors)
print("OCR empty results:", ocr_empty_results)
print("OCR rejected by confidence:", ocr_rejected_confidence)
print("OCR rejected by validation:", ocr_rejected_validation)
print("OCR accepted:", ocr_accepted)
print("OCR success rate:", round((ocr_successes / max(ocr_attempts, 1)) * 100.0, 2), "%")
print("Valid frame reads:", valid_frame_readings)
print("Final plate:", reported_plate)
print("Consensus plate:", stable_plate if stable_plate else "NONE")
if final_reason:
    print("Final plate reason:", final_reason)
print("Stable votes:", stable_votes)
print("Stability:", round(float(stability), 2), "%")
print("Average YOLO time:", round((total_yolo_seconds / max(yolo_runs, 1)) * 1000.0, 2), "ms")
print("Average preprocessing time:", round((total_preprocess_seconds / max(ocr_completed, 1)) * 1000.0, 2), "ms")
print("Average OCR time:", round((total_ocr_seconds / max(ocr_completed, 1)) * 1000.0, 2), "ms")
print("Average frame processing time:", round((total_frame_process_seconds / max(total_frames, 1)) * 1000.0, 2), "ms")
print("FPS:", round(float(processing_fps), 2))
print("Validation:", status)
print("Debug crops saved:", debug_crop_count)
print("=" * 70)
print(f"JSON result: {FINAL_JSON}")
print("=" * 70)
