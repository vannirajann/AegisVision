import cv2
import re
import time
from pathlib import Path
from collections import Counter, deque

from ultralytics import YOLO
from paddleocr import PaddleOCR


# ============================================================
# AEGISVISION - ANPR
# FINAL IMPROVED MAIN.PY
# ============================================================

print("=" * 65)
print("AEGISVISION - ANPR")
print("VIDEO NUMBER PLATE RECOGNITION")
print("=" * 65)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Your actual trained YOLO model
MODEL_PATH = BASE_DIR / "models" / "plate_detector.pt"

# Input folder
INPUT_DIR = BASE_DIR / "input"

# Output folder
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

# YOLO plate detection confidence
YOLO_CONFIDENCE = 0.35

# OCR every N frames
# 5 gives a good balance between speed and OCR stability
OCR_INTERVAL = 5

# Number of recent OCR results used
VOTING_WINDOW = 20

# Minimum repeated result required
MIN_VOTES = 3

# OCR confidence threshold
MIN_OCR_CONFIDENCE = 0.35

# Crop padding
PADDING_RATIO = 0.08

# Resize plate before OCR
UPSCALE_FACTOR = 4

# Minimum plate width
MIN_PLATE_WIDTH = 20

# Minimum plate height
MIN_PLATE_HEIGHT = 8


# ============================================================
# FIND VIDEO
# ============================================================

def find_video():

    video_extensions = [
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm"
    ]

    # First search input folder
    if INPUT_DIR.exists():

        videos = [
            file
            for file in INPUT_DIR.rglob("*")
            if file.suffix.lower() in video_extensions
        ]

        if videos:

            return videos[0]

    # Then search project folder
    videos = [
        file
        for file in BASE_DIR.rglob("*")
        if file.suffix.lower() in video_extensions
        and "venv" not in file.parts
    ]

    if videos:

        return videos[0]

    return None


# ============================================================
# CHECK MODEL
# ============================================================

print("\n[1] Checking YOLO model...")

if not MODEL_PATH.exists():

    print("\nERROR: YOLO model not found.")
    print("Expected:")
    print(MODEL_PATH)

    print("\nAvailable .pt files:")

    pt_files = list(
        BASE_DIR.rglob("*.pt")
    )

    if pt_files:

        for file in pt_files:
            print(" -", file)

    else:

        print("No .pt files found.")

    raise SystemExit


print(
    f"Model found: {MODEL_PATH}"
)


# ============================================================
# LOAD YOLO
# ============================================================

print("\n[2] Loading YOLO model...")

try:

    model = YOLO(
        str(MODEL_PATH)
    )

    print(
        "YOLO loaded successfully."
    )

except Exception as error:

    print(
        "\nERROR loading YOLO:"
    )

    print(error)

    raise SystemExit


# ============================================================
# LOAD PADDLE OCR
# ============================================================

print("\n[3] Loading PaddleOCR...")

try:

    ocr = PaddleOCR(
        lang="en"
    )

    print(
        "PaddleOCR loaded successfully."
    )

except Exception as error:

    print(
        "\nERROR loading PaddleOCR:"
    )

    print(error)

    raise SystemExit


# ============================================================
# FIND VIDEO
# ============================================================

print("\n[4] Searching for input video...")

VIDEO_PATH = find_video()

if VIDEO_PATH is None:

    print("\nERROR: No video found.")

    print(
        "\nPut your video inside:"
    )

    print(
        INPUT_DIR
    )

    print(
        "\nSupported formats:"
    )

    print(
        ".mp4 .avi .mov .mkv .webm"
    )

    raise SystemExit


print(
    f"Video found: {VIDEO_PATH}"
)


# ============================================================
# OCR CLEANING
# ============================================================

def clean_text(text):

    if text is None:
        return ""

    text = str(text)

    text = text.upper()

    # Common OCR mistakes
    text = text.replace(" ", "")
    text = text.replace("-", "")
    text = text.replace("_", "")

    # Remove everything except letters/numbers
    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text


# ============================================================
# INDIAN PLATE VALIDATION
# ============================================================

def is_valid_plate(text):

    text = clean_text(text)

    if not text:
        return False

    patterns = [

        # TN10AB1234
        r"^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{1,4}$",

        # TN1AB1234
        r"^[A-Z]{2}[0-9][A-Z]{1,3}[0-9]{1,4}$",

        # TN10A1234
        r"^[A-Z]{2}[0-9]{2}[A-Z][0-9]{1,4}$",

        # TN10AB123
        r"^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{1,4}$",

    ]

    for pattern in patterns:

        if re.fullmatch(
            pattern,
            text
        ):

            return True

    return False


# ============================================================
# POSSIBLE OCR RESULT
# ============================================================

def is_possible_plate(text):

    text = clean_text(text)

    if not text:
        return False

    if len(text) < 5:
        return False

    if len(text) > 12:
        return False

    # Must contain at least one letter
    if not re.search(
        r"[A-Z]",
        text
    ):
        return False

    # Must contain at least one number
    if not re.search(
        r"[0-9]",
        text
    ):
        return False

    return True


# ============================================================
# PLATE CROP
# ============================================================

def crop_plate(
    frame,
    x1,
    y1,
    x2,
    y2
):

    height, width = frame.shape[:2]

    plate_width = x2 - x1
    plate_height = y2 - y1

    if plate_width < MIN_PLATE_WIDTH:
        return None

    if plate_height < MIN_PLATE_HEIGHT:
        return None

    pad_x = int(
        plate_width *
        PADDING_RATIO
    )

    pad_y = int(
        plate_height *
        PADDING_RATIO
    )

    x1 = max(
        0,
        x1 - pad_x
    )

    y1 = max(
        0,
        y1 - pad_y
    )

    x2 = min(
        width,
        x2 + pad_x
    )

    y2 = min(
        height,
        y2 + pad_y
    )

    crop = frame[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:
        return None

    return crop


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_plate(plate):

    if plate is None:
        return []

    if plate.size == 0:
        return []

    # --------------------------------------------------------
    # UPSCALE
    # --------------------------------------------------------

    enlarged = cv2.resize(
        plate,
        None,
        fx=UPSCALE_FACTOR,
        fy=UPSCALE_FACTOR,
        interpolation=cv2.INTER_CUBIC
    )

    # --------------------------------------------------------
    # SHARPEN
    # --------------------------------------------------------

    blur = cv2.GaussianBlur(
        enlarged,
        (0, 0),
        2
    )

    sharpened = cv2.addWeighted(
        enlarged,
        1.4,
        blur,
        -0.4,
        0
    )

    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        sharpened,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        gray
    )

    # --------------------------------------------------------
    # OTSU
    # --------------------------------------------------------

    _, otsu = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )

    # --------------------------------------------------------
    # ADAPTIVE
    # --------------------------------------------------------

    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return [
        enlarged,
        sharpened,
        enhanced,
        otsu,
        adaptive
    ]


# ============================================================
# EXTRACT OCR RESULTS
# ============================================================

def extract_ocr_results(result):

    results = []

    try:

        # PaddleOCR 3.x result object
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

                import json

                data = json.loads(
                    data
                )

            if isinstance(
                data,
                dict
            ):

                texts = data.get(
                    "rec_texts",
                    []
                )

                scores = data.get(
                    "rec_scores",
                    []
                )

                for index, text in enumerate(
                    texts
                ):

                    score = 1.0

                    if index < len(scores):

                        try:

                            score = float(
                                scores[index]
                            )

                        except Exception:

                            score = 1.0

                    results.append(
                        (
                            str(text),
                            score
                        )
                    )

        # Some PaddleOCR versions expose
        # rec_texts / rec_scores directly
        elif hasattr(
            result,
            "rec_texts"
        ):

            texts = result.rec_texts

            scores = getattr(
                result,
                "rec_scores",
                []
            )

            for index, text in enumerate(
                texts
            ):

                score = 1.0

                if index < len(scores):

                    try:

                        score = float(
                            scores[index]
                        )

                    except Exception:

                        score = 1.0

                results.append(
                    (
                        str(text),
                        score
                    )
                )

    except Exception:
        pass

    return results


# ============================================================
# RUN OCR
# ============================================================

def run_ocr(image):

    all_results = []

    try:

        output = ocr.predict(
            image
        )

        if output is None:
            return all_results

        for result in output:

            extracted = extract_ocr_results(
                result
            )

            all_results.extend(
                extracted
            )

    except Exception as error:

        print(
            f"OCR warning: {error}"
        )

    return all_results


# ============================================================
# GET BEST PLATE FROM OCR
# ============================================================

def extract_best_plate(
    plate
):

    images = preprocess_plate(
        plate
    )

    if not images:
        return None, 0.0

    candidates = []

    for image in images:

        results = run_ocr(
            image
        )

        for text, score in results:

            cleaned = clean_text(
                text
            )

            if not cleaned:
                continue

            if score < MIN_OCR_CONFIDENCE:
                continue

            # ------------------------------------------------
            # FULL VALID PLATE
            # ------------------------------------------------

            if is_valid_plate(
                cleaned
            ):

                candidates.append(
                    (
                        cleaned,
                        score,
                        2
                    )
                )

            # ------------------------------------------------
            # POSSIBLE PARTIAL PLATE
            # ------------------------------------------------

            elif is_possible_plate(
                cleaned
            ):

                candidates.append(
                    (
                        cleaned,
                        score,
                        1
                    )
                )

    if not candidates:

        return None, 0.0

    # Valid plate > confidence > length
    candidates.sort(
        key=lambda item: (
            item[2],
            item[1],
            len(item[0])
        ),
        reverse=True
    )

    return (
        candidates[0][0],
        candidates[0][1]
    )


# ============================================================
# TEMPORAL VOTING
# ============================================================

plate_history = deque(
    maxlen=VOTING_WINDOW
)


# ============================================================
# ADD OCR RESULT
# ============================================================

def add_history(text):

    if text is None:
        return

    text = clean_text(
        text
    )

    if not text:
        return

    plate_history.append(
        text
    )


# ============================================================
# SIMILARITY BETWEEN OCR RESULTS
# ============================================================

def character_similarity(
    a,
    b
):

    a = clean_text(a)
    b = clean_text(b)

    if not a or not b:
        return 0.0

    # Same text
    if a == b:
        return 1.0

    # Simple positional similarity
    max_len = max(
        len(a),
        len(b)
    )

    min_len = min(
        len(a),
        len(b)
    )

    same = 0

    for i in range(min_len):

        if a[i] == b[i]:
            same += 1

    return same / max_len


# ============================================================
# GET FINAL RESULT
# ============================================================

def get_final_plate():

    if not plate_history:

        return (
            "UNKNOWN",
            0
        )

    counts = Counter(
        plate_history
    )

    # --------------------------------------------------------
    # First check exact votes
    # --------------------------------------------------------

    best_plate, votes = (
        counts.most_common(1)[0]
    )

    if is_valid_plate(
        best_plate
    ):

        if votes >= MIN_VOTES:

            return (
                best_plate,
                votes
            )

    # --------------------------------------------------------
    # Try grouping similar OCR results
    # --------------------------------------------------------

    history = list(
        plate_history
    )

    groups = []

    for text in history:

        added = False

        for group in groups:

            representative = group[0]

            similarity = (
                character_similarity(
                    text,
                    representative
                )
            )

            if similarity >= 0.70:

                group.append(
                    text
                )

                added = True

                break

        if not added:

            groups.append(
                [text]
            )

    # --------------------------------------------------------
    # Find strongest group
    # --------------------------------------------------------

    best_group = None

    for group in groups:

        if (
            best_group is None
            or len(group) > len(best_group)
        ):

            best_group = group

    if best_group:

        group_counts = Counter(
            best_group
        )

        candidate, candidate_votes = (
            group_counts.most_common(1)[0]
        )

        if is_valid_plate(
            candidate
        ):

            if candidate_votes >= MIN_VOTES:

                return (
                    candidate,
                    candidate_votes
                )

    return (
        "UNKNOWN",
        votes
    )


# ============================================================
# OPEN VIDEO
# ============================================================

print("\n[5] Opening video...")

cap = cv2.VideoCapture(
    str(VIDEO_PATH)
)

if not cap.isOpened():

    print(
        "\nERROR: Cannot open video."
    )

    print(
        VIDEO_PATH
    )

    raise SystemExit


# ============================================================
# VIDEO INFORMATION
# ============================================================

total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)

source_fps = cap.get(
    cv2.CAP_PROP_FPS
)

width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

print(
    f"Resolution : {width} x {height}"
)

print(
    f"Video FPS  : {source_fps:.2f}"
)

print(
    f"Frames     : {total_frames}"
)

print(
    f"OCR        : Every {OCR_INTERVAL} frames"
)

print("\nStarting ANPR...")
print("Press Q to stop.\n")


# ============================================================
# OUTPUT VIDEO
# ============================================================

output_video_path = (
    OUTPUT_DIR /
    "anpr_result.mp4"
)

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    str(output_video_path),
    fourcc,
    source_fps if source_fps > 0 else 25,
    (width, height)
)


# ============================================================
# VARIABLES
# ============================================================

frame_count = 0

start_time = time.time()

fps = 0.0

last_box = None

last_detection_confidence = 0.0

last_ocr = "None"

last_ocr_confidence = 0.0

last_final_plate = "UNKNOWN"

last_votes = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        break

    frame_count += 1


    # ========================================================
    # YOLO
    # ========================================================

    try:

        results = model.predict(
            frame,
            conf=YOLO_CONFIDENCE,
            verbose=False
        )

    except Exception as error:

        print(
            f"YOLO error at frame "
            f"{frame_count}: {error}"
        )

        continue


    detections = []


    # ========================================================
    # READ YOLO BOXES
    # ========================================================

    for result in results:

        if result.boxes is None:

            continue

        for box in result.boxes:

            try:

                coords = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                )

                x1, y1, x2, y2 = map(
                    int,
                    coords
                )

                confidence = float(
                    box.conf[0]
                )

                if x2 <= x1:
                    continue

                if y2 <= y1:
                    continue

                detections.append(
                    (
                        x1,
                        y1,
                        x2,
                        y2,
                        confidence
                    )
                )

            except Exception:

                continue


    # ========================================================
    # SELECT BEST DETECTION
    # ========================================================

    if detections:

        detections.sort(
            key=lambda item: (
                item[4],
                (item[2] - item[0]) *
                (item[3] - item[1])
            ),
            reverse=True
        )

        (
            x1,
            y1,
            x2,
            y2,
            detection_confidence
        ) = detections[0]

        last_box = (
            x1,
            y1,
            x2,
            y2
        )

        last_detection_confidence = (
            detection_confidence
        )


        # ====================================================
        # OCR
        # ====================================================

        if (
            frame_count %
            OCR_INTERVAL
            == 0
        ):

            plate = crop_plate(
                frame,
                x1,
                y1,
                x2,
                y2
            )

            if plate is not None:

                text, confidence = (
                    extract_best_plate(
                        plate
                    )
                )

                if text:

                    last_ocr = text

                    last_ocr_confidence = (
                        confidence
                    )

                    add_history(
                        text
                    )

                    print(
                        f"Frame {frame_count:4d} | "
                        f"OCR: {text:<15} | "
                        f"Confidence: {confidence:.2f}"
                    )

                    final_plate, votes = (
                        get_final_plate()
                    )

                    last_final_plate = (
                        final_plate
                    )

                    last_votes = votes


    # ========================================================
    # FPS
    # ========================================================

    elapsed = (
        time.time() -
        start_time
    )

    if elapsed > 0:

        fps = (
            frame_count /
            elapsed
        )


    # ========================================================
    # DRAW PLATE BOX
    # ========================================================

    if last_box is not None:

        x1, y1, x2, y2 = (
            last_box
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Plate Detection "
            f"{last_detection_confidence:.2f}",
            (
                x1,
                max(
                    25,
                    y1 - 8
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )


    # ========================================================
    # INFORMATION PANEL
    # ========================================================

    cv2.putText(
        frame,
        f"PLATE: {last_final_plate}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"OCR: {last_ocr}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"OCR Confidence: "
        f"{last_ocr_confidence:.2f}",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Votes: {last_votes}",
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Frame: "
        f"{frame_count}/{total_frames}",
        (20, 215),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SAVE OUTPUT FRAME
    # ========================================================

    writer.write(
        frame
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "AegisVision ANPR",
        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

writer.release()

cv2.destroyAllWindows()


# ============================================================
# FINAL RESULT
# ============================================================

final_plate, final_votes = (
    get_final_plate()
)

elapsed_total = (
    time.time() -
    start_time
)

average_fps = (
    frame_count /
    elapsed_total
    if elapsed_total > 0
    else 0
)


print("\n")
print("=" * 65)
print("AEGISVISION ANPR - COMPLETE")
print("=" * 65)

print(
    f"Frames processed : {frame_count}"
)

print(
    f"Average FPS      : {average_fps:.2f}"
)

print(
    f"OCR results      : {len(plate_history)}"
)

print(
    f"Last OCR         : {last_ocr}"
)

print(
    f"Final plate      : {final_plate}"
)

print(
    f"Final votes      : {final_votes}"
)

print(
    f"Output video     : {output_video_path}"
)

print("=" * 65)

if final_plate == "UNKNOWN":

    print(
        "FINAL RESULT: UNKNOWN"
    )

    print(
        "No sufficiently reliable plate "
        "was confirmed."
    )

else:

    print(
        f"FINAL RESULT: {final_plate}"
    )

print("=" * 65)