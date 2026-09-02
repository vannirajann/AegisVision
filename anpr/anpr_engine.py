# ============================================================
# AEGISVISION - ANPR
# PHASE 5.2 - ANPR ENGINE
#
# Purpose:
# Reusable ANPR processing engine for API integration.
#
# Pipeline:
#
# Image
#   ↓
# YOLO Plate Detection
#   ↓
# Plate Crop
#   ↓
# Image Preprocessing
#   ↓
# Tesseract OCR
#   ↓
# OCR Correction
#   ↓
# Indian Plate Validation
#   ↓
# Confidence Evaluation
#   ↓
# Structured ANPR Result
#
# ============================================================

import os
import re
from pathlib import Path

import cv2
import pytesseract
from ultralytics import YOLO


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "plate_detector.pt"


# ============================================================
# SETTINGS
# ============================================================

YOLO_CONFIDENCE = 0.25

OCR_CONFIDENCE_THRESHOLD = 20.0

TESSERACT_PATH = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# ============================================================
# OCR CHARACTER CORRECTION
# ============================================================

LETTER_TO_DIGIT = {
    "O": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6"
}


# ============================================================
# LOAD YOLO MODEL
# ============================================================

_model = None


def load_model():
    """
    Load YOLO plate detection model.

    The model is loaded only once and then reused.
    """

    global _model

    if _model is None:

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"YOLO model not found: {MODEL_PATH}"
            )

        _model = YOLO(str(MODEL_PATH))

    return _model


# ============================================================
# CONFIGURE TESSERACT
# ============================================================

def configure_tesseract():
    """
    Configure Tesseract OCR.
    """

    if os.path.exists(TESSERACT_PATH):

        pytesseract.pytesseract.tesseract_cmd = (
            TESSERACT_PATH
        )

        return True

    try:

        pytesseract.get_tesseract_version()

        return True

    except Exception:

        return False


# ============================================================
# CLEAN OCR TEXT
# ============================================================

def clean_plate_text(text):
    """
    Remove spaces and unwanted OCR characters.
    """

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
# NORMALIZE OCR TEXT
# ============================================================

def normalize_ocr_characters(text):
    """
    Normalize OCR output.
    """

    if not text:
        return ""

    return (
        str(text)
        .upper()
        .strip()
    )


# ============================================================
# INDIAN PLATE VALIDATION
# ============================================================

def validate_indian_plate(text):
    """
    Validate standard Indian vehicle number plate structure.

    Examples:

    TN09BY9726
    KA01AB1234
    MH12CD1234
    """

    if not text:
        return False

    pattern = (
        r"^[A-Z]{2}"
        r"[0-9]{1,2}"
        r"[A-Z]{1,3}"
        r"[0-9]{1,4}$"
    )

    return bool(
        re.fullmatch(
            pattern,
            text
        )
    )


# ============================================================
# GENERATE OCR CORRECTION CANDIDATES
# ============================================================

def generate_ocr_candidates(text):
    """
    Generate possible corrected plate numbers.
    """

    if not text:
        return []

    text = clean_plate_text(text)

    if not text:
        return []

    candidates = []

    # Original OCR text
    candidates.append(text)

    # --------------------------------------------------------
    # Try district length 1 or 2 digits
    # --------------------------------------------------------

    for district_length in (1, 2):

        district_start = 2

        district_end = (
            district_start
            + district_length
        )

        if len(text) <= district_end:
            continue

        chars = list(text)

        valid_digit_section = True

        # ----------------------------------------------------
        # Correct district characters
        # ----------------------------------------------------

        for index in range(
            district_start,
            district_end
        ):

            character = chars[index]

            if character.isdigit():
                continue

            if character in LETTER_TO_DIGIT:

                chars[index] = (
                    LETTER_TO_DIGIT[character]
                )

            else:

                valid_digit_section = False

        if not valid_digit_section:
            continue

        corrected = "".join(chars)

        if corrected not in candidates:

            candidates.append(corrected)

        # ----------------------------------------------------
        # Try series length 1, 2, 3
        # ----------------------------------------------------

        for series_length in (
            1,
            2,
            3
        ):

            number_start = (
                district_end
                + series_length
            )

            if number_start >= len(text):
                continue

            chars2 = list(corrected)

            valid_number_section = True

            # ------------------------------------------------
            # Correct final number section
            # ------------------------------------------------

            for index in range(
                number_start,
                len(chars2)
            ):

                character = chars2[index]

                if character.isdigit():
                    continue

                if character in LETTER_TO_DIGIT:

                    chars2[index] = (
                        LETTER_TO_DIGIT[character]
                    )

                else:

                    valid_number_section = False

            if not valid_number_section:
                continue

            candidate = "".join(chars2)

            if candidate not in candidates:

                candidates.append(candidate)

    return candidates


# ============================================================
# CORRECT OCR PLATE
# ============================================================

def correct_ocr_plate(text):
    """
    Correct OCR mistakes when a candidate
    matches the Indian plate format.
    """

    original = clean_plate_text(text)

    if not original:

        return (
            "",
            False,
            []
        )

    candidates = (
        generate_ocr_candidates(
            original
        )
    )

    # --------------------------------------------------------
    # Original is already valid
    # --------------------------------------------------------

    if validate_indian_plate(original):

        return (
            original,
            False,
            candidates
        )

    # --------------------------------------------------------
    # Search corrected candidates
    # --------------------------------------------------------

    for candidate in candidates:

        if validate_indian_plate(candidate):

            return (
                candidate,
                candidate != original,
                candidates
            )

    return (
        original,
        False,
        candidates
    )


# ============================================================
# PLATE PREPROCESSING
# ============================================================

def preprocess_plate(crop):
    """
    Generate multiple preprocessing versions
    of a detected number plate.
    """

    if crop is None:
        return {}

    if crop.size == 0:
        return {}

    height, width = crop.shape[:2]

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    new_width = max(
        width * 5,
        300
    )

    new_height = max(
        height * 5,
        100
    )

    resized = cv2.resize(
        crop,
        (
            new_width,
            new_height
        ),
        interpolation=cv2.INTER_CUBIC
    )

    # --------------------------------------------------------
    # Grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Normal
    # --------------------------------------------------------

    normal = gray.copy()

    # --------------------------------------------------------
    # Equalized
    # --------------------------------------------------------

    equalized = cv2.equalizeHist(
        gray
    )

    # --------------------------------------------------------
    # Blur
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        equalized,
        (3, 3),
        0
    )

    # --------------------------------------------------------
    # OTSU
    # --------------------------------------------------------

    _, otsu = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU
    )

    # --------------------------------------------------------
    # Adaptive threshold
    # --------------------------------------------------------

    threshold = cv2.adaptiveThreshold(
        equalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    # --------------------------------------------------------
    # Sharpen
    # --------------------------------------------------------

    blurred2 = cv2.GaussianBlur(
        equalized,
        (0, 0),
        3
    )

    sharpened = cv2.addWeighted(
        equalized,
        1.5,
        blurred2,
        -0.5,
        0
    )

    return {
        "normal": normal,
        "equalized": equalized,
        "threshold": threshold,
        "otsu": otsu,
        "sharpened": sharpened
    }


# ============================================================
# TESSERACT OCR
# ============================================================

def run_tesseract(image):
    """
    Run Tesseract OCR using multiple PSM configurations.
    """

    if image is None:
        return "", 0.0

    best_text = ""

    best_confidence = 0.0

    configurations = [

        (
            "--psm 7 "
            "-c tessedit_char_whitelist="
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        ),

        (
            "--psm 8 "
            "-c tessedit_char_whitelist="
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        ),

        (
            "--psm 6 "
            "-c tessedit_char_whitelist="
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

    ]

    for config in configurations:

        try:

            # ------------------------------------------------
            # OCR TEXT
            # ------------------------------------------------

            text = pytesseract.image_to_string(
                image,
                config=config
            )

            cleaned = clean_plate_text(
                text
            )

            cleaned = normalize_ocr_characters(
                cleaned
            )

            # ------------------------------------------------
            # OCR CONFIDENCE
            # ------------------------------------------------

            data = pytesseract.image_to_data(
                image,
                config=config,
                output_type=pytesseract.Output.DICT
            )

            confidences = []

            for value in data["conf"]:

                try:

                    value = float(value)

                    if value >= 0:

                        confidences.append(
                            value
                        )

                except (
                    ValueError,
                    TypeError
                ):

                    continue

            if confidences:

                confidence = (
                    sum(confidences)
                    / len(confidences)
                )

            else:

                confidence = 0.0

            # ------------------------------------------------
            # Select best OCR result
            # ------------------------------------------------

            if cleaned:

                if (
                    confidence > best_confidence
                    or not best_text
                ):

                    best_text = cleaned

                    best_confidence = (
                        confidence
                    )

        except Exception:

            continue

    return (
        best_text,
        best_confidence
    )


# ============================================================
# MULTI-PREPROCESS OCR
# ============================================================

def run_plate_ocr(plate_images):
    """
    Run OCR on all preprocessing versions.
    """

    best_text = ""

    best_confidence = 0.0

    best_method = ""

    for method, image in plate_images.items():

        text, confidence = run_tesseract(
            image
        )

        if not text:
            continue

        if (
            confidence > best_confidence
            or not best_text
        ):

            best_text = text

            best_confidence = confidence

            best_method = method

    return (
        best_text,
        best_confidence,
        best_method
    )


# ============================================================
# OCR QUALITY
# ============================================================

def get_ocr_quality(confidence):
    """
    Convert OCR confidence into a readable quality label.
    """

    if confidence >= 70:

        return "HIGH"

    elif confidence >= OCR_CONFIDENCE_THRESHOLD:

        return "ACCEPTABLE"

    else:

        return "LOW"


# ============================================================
# FINAL OCR CLASSIFICATION
# ============================================================

def classify_ocr_result(
    plate_text,
    confidence
):
    """
    Classification:

    VALID:
        Valid Indian plate format
        AND acceptable OCR confidence.

    REVIEW:
        Valid Indian plate format
        BUT low OCR confidence.

    INVALID:
        OCR text exists but does not
        match Indian plate structure.

    FAILED:
        No OCR text.
    """

    if not plate_text:

        return "FAILED"

    valid = validate_indian_plate(
        plate_text
    )

    if not valid:

        return "INVALID"

    if confidence >= OCR_CONFIDENCE_THRESHOLD:

        return "VALID"

    return "REVIEW"


# ============================================================
# OVERALL CONFIDENCE
# ============================================================

def calculate_overall_confidence(
    detection_confidence,
    ocr_confidence
):
    """
    Calculate combined ANPR confidence.

    Detection confidence is converted from
    0-1 into 0-100.

    Final confidence is weighted:

        60% detection
        40% OCR
    """

    detection_percent = (
        detection_confidence * 100
    )

    overall = (
        detection_percent * 0.60
        + ocr_confidence * 0.40
    )

    return round(
        overall,
        2
    )


# ============================================================
# PROCESS SINGLE PLATE
# ============================================================

def process_plate(
    image,
    bbox,
    detection_confidence
):
    """
    Process one detected number plate.
    """

    x1, y1, x2, y2 = bbox

    crop = image[
        y1:y2,
        x1:x2
    ]

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    plate_images = preprocess_plate(
        crop
    )

    if not plate_images:

        return {
            "plate_number": "",
            "raw_ocr_text": "",
            "cleaned_ocr_text": "",
            "corrected_ocr_text": "",
            "correction_applied": False,
            "correction_candidates": [],
            "status": "FAILED",
            "result": {
                "valid_indian_plate": False,
                "message": "INVALID PLATE CROP"
            },
            "confidence": {
                "detection": round(
                    detection_confidence,
                    4
                ),
                "ocr": 0.0,
                "overall": round(
                    detection_confidence * 60,
                    2
                )
            },
            "ocr_quality": "LOW",
            "ocr_method": "",
            "bounding_box": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            }
        }

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    (
        raw_ocr,
        ocr_confidence,
        ocr_method
    ) = run_plate_ocr(
        plate_images
    )

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    cleaned_text = clean_plate_text(
        raw_ocr
    )

    # --------------------------------------------------------
    # Correction
    # --------------------------------------------------------

    (
        corrected_text,
        correction_applied,
        candidates
    ) = correct_ocr_plate(
        cleaned_text
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    classification = classify_ocr_result(
        corrected_text,
        ocr_confidence
    )

    # --------------------------------------------------------
    # Overall confidence
    # --------------------------------------------------------

    overall_confidence = calculate_overall_confidence(
        detection_confidence,
        ocr_confidence
    )

    # --------------------------------------------------------
    # Result message
    # --------------------------------------------------------

    if classification == "VALID":

        message = "VALID PLATE"

    elif classification == "REVIEW":

        message = "LOW CONFIDENCE - REVIEW"

    elif classification == "INVALID":

        message = "INVALID PLATE FORMAT"

    else:

        message = "OCR FAILED"

    # --------------------------------------------------------
    # Return structured result
    # --------------------------------------------------------

    return {
        "plate_number": corrected_text,
        "raw_ocr_text": raw_ocr,
        "cleaned_ocr_text": cleaned_text,
        "corrected_ocr_text": corrected_text,
        "correction_applied": correction_applied,
        "correction_candidates": candidates,

        "status": classification,

        "result": {
            "valid_indian_plate":
                validate_indian_plate(
                    corrected_text
                ),
            "message": message
        },

        "confidence": {
            "detection":
                round(
                    detection_confidence,
                    4
                ),

            "ocr":
                round(
                    ocr_confidence,
                    2
                ),

            "overall":
                overall_confidence
        },

        "ocr_quality":
            get_ocr_quality(
                ocr_confidence
            ),

        "ocr_method":
            ocr_method,

        "bounding_box": {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        }
    }


# ============================================================
# MAIN ANPR ENGINE
# ============================================================

def process_image(image):
    """
    Main reusable ANPR function.

    Input:
        OpenCV BGR image.

    Output:
        Structured Python dictionary.
    """

    if image is None:

        return {
            "success": False,
            "message": "Invalid image",
            "plate_count": 0,
            "plates": []
        }

    # --------------------------------------------------------
    # Configure Tesseract
    # --------------------------------------------------------

    if not configure_tesseract():

        return {
            "success": False,
            "message": "Tesseract OCR not available",
            "plate_count": 0,
            "plates": []
        }

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    try:

        model = load_model()

    except Exception as error:

        return {
            "success": False,
            "message": str(error),
            "plate_count": 0,
            "plates": []
        }

    # --------------------------------------------------------
    # Image information
    # --------------------------------------------------------

    height, width = image.shape[:2]

    # --------------------------------------------------------
    # YOLO detection
    # --------------------------------------------------------

    try:

        results = model.predict(
            source=image,
            conf=YOLO_CONFIDENCE,
            verbose=False
        )

    except Exception as error:

        return {
            "success": False,
            "message":
                f"YOLO detection failed: {error}",
            "plate_count": 0,
            "plates": []
        }

    # --------------------------------------------------------
    # Collect detections
    # --------------------------------------------------------

    detections = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            detection_confidence = float(
                box.conf[0]
            )

            coordinates = (
                box.xyxy[0].tolist()
            )

            x1, y1, x2, y2 = map(
                int,
                coordinates
            )

            # ------------------------------------------------
            # Keep coordinates inside image
            # ------------------------------------------------

            x1 = max(
                0,
                min(
                    x1,
                    width - 1
                )
            )

            y1 = max(
                0,
                min(
                    y1,
                    height - 1
                )
            )

            x2 = max(
                0,
                min(
                    x2,
                    width
                )
            )

            y2 = max(
                0,
                min(
                    y2,
                    height
                )
            )

            if x2 <= x1:
                continue

            if y2 <= y1:
                continue

            detections.append(
                {
                    "confidence":
                        detection_confidence,

                    "bbox": [
                        x1,
                        y1,
                        x2,
                        y2
                    ]
                }
            )

    # --------------------------------------------------------
    # No plates
    # --------------------------------------------------------

    if not detections:

        return {
            "success": True,

            "message":
                "No number plate detected",

            "image_size": {
                "width": width,
                "height": height
            },

            "plate_count": 0,

            "plates": []
        }

    # --------------------------------------------------------
    # Process plates
    # --------------------------------------------------------

    plates = []

    for plate_index, detection in enumerate(
        detections,
        start=1
    ):

        result = process_plate(
            image=image,
            bbox=detection["bbox"],
            detection_confidence=
                detection["confidence"]
        )

        result["plate_id"] = plate_index

        # ----------------------------------------------------
        # Keep plate_id first in dictionary
        # ----------------------------------------------------

        result = {
            "plate_id":
                plate_index,

            **{
                key: value
                for key, value
                in result.items()
                if key != "plate_id"
            }
        }

        plates.append(result)

    # --------------------------------------------------------
    # Final structured response
    # --------------------------------------------------------

    return {
        "success": True,

        "message":
            "ANPR processing completed",

        "image_size": {
            "width": width,
            "height": height
        },

        "plate_count":
            len(plates),

        "plates":
            plates
    }


# ============================================================
# PROCESS IMAGE FILE
# ============================================================

def process_image_file(image_path):
    """
    Convenience function for testing an image file.
    """

    image_path = Path(image_path)

    if not image_path.exists():

        return {
            "success": False,
            "message":
                f"Image not found: {image_path}",
            "plate_count": 0,
            "plates": []
        }

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        return {
            "success": False,
            "message":
                "Could not read image",
            "plate_count": 0,
            "plates": []
        }

    return process_image(
        image
    )


# ============================================================
# ENGINE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "AEGISVISION - ANPR ENGINE"
    )

    print(
        "PHASE 5.2 ENGINE TEST"
    )

    print("=" * 60)

    test_image = (
        BASE_DIR
        / "input"
        / "car.jpg"
    )

    print(
        f"\nTesting image:"
    )

    print(
        test_image
    )

    result = process_image_file(
        test_image
    )

    print("\nANPR RESULT")
    print("-" * 60)

    import json

    print(
        json.dumps(
            result,
            indent=4
        )
    )

    print("\n" + "=" * 60)

    if result.get("success"):

        print(
            "ANPR ENGINE TEST PASSED"
        )

    else:

        print(
            "ANPR ENGINE TEST FAILED"
        )

    print("=" * 60)