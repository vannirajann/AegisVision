"""
============================================================
AEGISVISION - ANPR
PHASE 5.7 - OCR ACCURACY IMPROVEMENT
OCR NORMALIZATION + MULTI-READING CONSISTENCY
============================================================

Input:
    output/phase5_5/phase5_5_report.json

Output:
    output/phase5_7/phase5_7_report.json
    output/phase5_7/accuracy_results.csv

This phase does NOT modify the original Phase 5.5 report.
"""

import json
import csv
import re
from pathlib import Path
from collections import Counter


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_REPORT = (
    BASE_DIR
    / "output"
    / "phase5_5"
    / "phase5_5_report.json"
)

OUTPUT_DIR = (
    BASE_DIR
    / "output"
    / "phase5_7"
)

OUTPUT_JSON = OUTPUT_DIR / "phase5_7_report.json"
OUTPUT_CSV = OUTPUT_DIR / "accuracy_results.csv"


# ============================================================
# CONFIGURATION
# ============================================================

MIN_LENGTH = 6
MAX_LENGTH = 12

# Confidence levels
VALID_CONFIDENCE = 60.0
REVIEW_CONFIDENCE = 35.0

# Minimum number of observations for a strong multi-frame result
STRONG_OBSERVATIONS = 3

# Character corrections used only for comparison.
# We do NOT blindly replace every character in the final plate.
OCR_CONFUSIONS = {
    "0": ["O", "Q", "D"],
    "O": ["0", "Q", "D"],
    "1": ["I", "L"],
    "I": ["1", "L"],
    "L": ["1", "I"],
    "5": ["S"],
    "S": ["5"],
    "8": ["B"],
    "B": ["8"],
    "2": ["Z"],
    "Z": ["2"],
    "6": ["G"],
    "G": ["6"],
}


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("AEGISVISION - ANPR")
print("PHASE 5.7 - OCR ACCURACY IMPROVEMENT")
print("OCR NORMALIZATION + MULTI-READING CONSISTENCY")
print("=" * 60)

print()
print("Input report:")
print(INPUT_REPORT)


# ============================================================
# CHECK INPUT
# ============================================================

if not INPUT_REPORT.exists():
    print()
    print("ERROR: Phase 5.5 report not found.")
    print()
    print("Expected:")
    print(INPUT_REPORT)
    print()
    print("Run Phase 5.5 first.")
    raise SystemExit(1)


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD JSON
# ============================================================

try:
    with open(INPUT_REPORT, "r", encoding="utf-8") as file:
        report = json.load(file)
except Exception as exc:
    print()
    print("ERROR: Could not read JSON report.")
    print(exc)
    raise SystemExit(1)


# ============================================================
# OCR CLEANING
# ============================================================

def clean_ocr(text):
    """
    Basic OCR text cleaning.

    Removes:
        spaces
        hyphens
        punctuation
        lowercase

    Keeps:
        A-Z
        0-9
    """

    if text is None:
        return ""

    text = str(text).upper().strip()

    # Remove spaces and common separators
    text = text.replace(" ", "")
    text = text.replace("-", "")
    text = text.replace("_", "")

    # Keep only letters and numbers
    text = re.sub(r"[^A-Z0-9]", "", text)

    return text


# ============================================================
# SIMILARITY
# ============================================================

def levenshtein_distance(a, b):
    """
    Calculate Levenshtein edit distance.
    """

    if a == b:
        return 0

    if not a:
        return len(b)

    if not b:
        return len(a)

    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):

        current = [i]

        for j, char_b in enumerate(b, start=1):

            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (char_a != char_b)

            current.append(
                min(
                    insert_cost,
                    delete_cost,
                    replace_cost
                )
            )

        previous = current

    return previous[-1]


def similarity(a, b):
    """
    Return similarity from 0 to 100.
    """

    if not a or not b:
        return 0.0

    distance = levenshtein_distance(a, b)

    maximum_length = max(len(a), len(b))

    if maximum_length == 0:
        return 100.0

    return round(
        (1.0 - distance / maximum_length) * 100.0,
        2
    )


# ============================================================
# OCR CHARACTER COMPATIBILITY
# ============================================================

def character_compatible(a, b):
    """
    Returns True if characters are identical or commonly confused
    by OCR.
    """

    if a == b:
        return True

    return b in OCR_CONFUSIONS.get(a, [])


def confusion_similarity(a, b):
    """
    Similarity that gives common OCR character confusions
    partial credit.
    """

    if not a or not b:
        return 0.0

    max_len = max(len(a), len(b))

    if max_len == 0:
        return 100.0

    score = 0

    for i in range(min(len(a), len(b))):

        if character_compatible(a[i], b[i]):
            score += 1

    # Penalize length difference
    length_penalty = abs(len(a) - len(b))

    effective_score = max(0, score - length_penalty)

    return round(
        (effective_score / max_len) * 100.0,
        2
    )


# ============================================================
# PLATE FORMAT CHECK
# ============================================================

def format_score(plate):
    """
    General Indian-style registration plate shape check.

    This is NOT an official RTO validation.
    It only checks whether the OCR result looks plausible.

    Examples of plausible structures:

        AB12CD1234
        TN01AB1234
        KA05MN1234
        DL8CAF1234

    Because state formats vary, this function is intentionally
    flexible.
    """

    if not plate:
        return 0.0

    length = len(plate)

    if length < MIN_LENGTH or length > MAX_LENGTH:
        return 0.0

    letters = sum(char.isalpha() for char in plate)
    digits = sum(char.isdigit() for char in plate)

    if letters == 0 or digits == 0:
        return 10.0

    score = 50.0

    # Most Indian plates contain both letters and numbers
    if letters >= 2:
        score += 15

    if digits >= 2:
        score += 15

    # Prefer lengths commonly encountered in registrations
    if 8 <= length <= 10:
        score += 20
    elif 6 <= length <= 11:
        score += 10

    return min(score, 100.0)


# ============================================================
# SELECT BEST OCR RESULT
# ============================================================

def select_best_candidate(readings, confidences=None):
    """
    Select the most reliable OCR candidate using:

        frequency
        OCR confidence
        similarity to other readings
        format plausibility
    """

    cleaned = []

    for reading in readings:

        value = clean_ocr(reading)

        if not value:
            continue

        if len(value) < MIN_LENGTH:
            continue

        if len(value) > MAX_LENGTH:
            continue

        cleaned.append(value)

    if not cleaned:
        return ""

    counts = Counter(cleaned)

    candidates = list(counts.keys())

    best_candidate = None
    best_score = -1.0

    for candidate in candidates:

        frequency = counts[candidate]

        # Frequency score
        frequency_score = min(
            100.0,
            frequency / max(len(cleaned), 1) * 100.0
        )

        # Similarity to all readings
        similarities = []

        for other in cleaned:
            similarities.append(
                confusion_similarity(candidate, other)
            )

        consistency_score = (
            sum(similarities) / len(similarities)
            if similarities
            else 0.0
        )

        # Format score
        plate_score = format_score(candidate)

        # Combined score
        score = (
            frequency_score * 0.40
            + consistency_score * 0.40
            + plate_score * 0.20
        )

        if score > best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate or ""


# ============================================================
# CALCULATE CONSISTENCY
# ============================================================

def calculate_consistency(final_plate, readings):
    """
    Calculate how consistently the OCR readings support
    the selected final plate.
    """

    if not final_plate or not readings:
        return 0.0

    scores = []

    for reading in readings:

        cleaned = clean_ocr(reading)

        if not cleaned:
            continue

        score = confusion_similarity(
            final_plate,
            cleaned
        )

        scores.append(score)

    if not scores:
        return 0.0

    return round(
        sum(scores) / len(scores),
        2
    )


# ============================================================
# FINAL DECISION
# ============================================================

def final_status(
    confidence,
    consistency,
    observations,
    plate
):
    """
    Determine final status.

    VALID:
        Strong confidence and enough evidence.

    REVIEW:
        Potentially correct but insufficient evidence.

    INVALID:
        Very weak result.
    """

    if not plate:
        return "INVALID"

    if len(plate) < MIN_LENGTH or len(plate) > MAX_LENGTH:
        return "INVALID"

    # Strong multi-frame evidence
    if (
        confidence >= VALID_CONFIDENCE
        and consistency >= 70
        and observations >= STRONG_OBSERVATIONS
    ):
        return "VALID"

    # Strong consistency with multiple observations
    if (
        consistency >= 80
        and observations >= 3
        and confidence >= 45
    ):
        return "VALID"

    # Moderate evidence
    if (
        confidence >= REVIEW_CONFIDENCE
        or consistency >= 45
        or observations >= 2
    ):
        return "REVIEW"

    return "INVALID"


# ============================================================
# EXTRACT TRACK RESULTS
# ============================================================

track_results = report.get("track_results", [])

if not track_results:
    # Fallback to final_plates
    track_results = report.get("final_plates", [])


print()
print("Extracting OCR readings...")
print(f"Tracks found: {len(track_results)}")


# ============================================================
# PROCESS TRACKS
# ============================================================

final_results = []

print()
print("PROCESSING OCR ACCURACY")
print("-" * 60)


for index, track in enumerate(track_results, start=1):

    track_id = str(
        track.get(
            "track_id",
            track.get("tracks", [index])[0]
            if isinstance(track.get("tracks", [index]), list)
            else track.get("tracks", index)
        )
    )

    original_plate = clean_ocr(
        track.get(
            "final_plate",
            track.get("plate", "")
        )
    )

    readings = track.get(
        "unique_ocr_readings",
        track.get("ocr_readings", [])
    )

    if not readings and original_plate:
        readings = [original_plate]

    cleaned_readings = []

    for reading in readings:

        value = clean_ocr(reading)

        if value:
            cleaned_readings.append(value)

    # Remove duplicates while preserving order
    cleaned_readings = list(
        dict.fromkeys(cleaned_readings)
    )

    observations = int(
        track.get(
            "observations",
            track.get(
                "total_observations",
                len(cleaned_readings)
            )
        )
    )

    votes = int(
        track.get(
            "votes",
            1
        )
    )

    original_confidence = float(
        track.get(
            "final_confidence",
            track.get(
                "confidence",
                track.get(
                    "average_confidence",
                    0.0
                )
            )
        )
    )

    # --------------------------------------------------------
    # Select improved candidate
    # --------------------------------------------------------

    improved_plate = select_best_candidate(
        cleaned_readings
    )

    if not improved_plate:
        improved_plate = original_plate

    # --------------------------------------------------------
    # Consistency
    # --------------------------------------------------------

    consistency = calculate_consistency(
        improved_plate,
        cleaned_readings
    )

    # --------------------------------------------------------
    # Format score
    # --------------------------------------------------------

    plate_format_score = format_score(
        improved_plate
    )

    # --------------------------------------------------------
    # Accuracy score
    # --------------------------------------------------------

    accuracy_score = (
        original_confidence * 0.45
        + consistency * 0.35
        + plate_format_score * 0.20
    )

    accuracy_score = round(
        min(100.0, accuracy_score),
        2
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    status = final_status(
        accuracy_score,
        consistency,
        observations,
        improved_plate
    )

    result = {
        "rank": 0,
        "track_id": track_id,
        "original_plate": original_plate,
        "improved_plate": improved_plate,
        "status": status,
        "original_confidence": round(
            original_confidence,
            2
        ),
        "consistency_score": consistency,
        "format_score": round(
            plate_format_score,
            2
        ),
        "accuracy_score": accuracy_score,
        "observations": observations,
        "votes": votes,
        "ocr_readings": cleaned_readings,
        "frames": track.get(
            "frames",
            []
        ),
    }

    final_results.append(result)

    print()
    print(f"Track {track_id}")
    print(f"  Original plate   : {original_plate}")
    print(f"  Improved plate   : {improved_plate}")
    print(f"  Status           : {status}")
    print(
        f"  Original conf.   : "
        f"{original_confidence:.2f}%"
    )
    print(
        f"  Consistency      : "
        f"{consistency:.2f}%"
    )
    print(
        f"  Format score     : "
        f"{plate_format_score:.2f}%"
    )
    print(
        f"  Accuracy score   : "
        f"{accuracy_score:.2f}%"
    )
    print(
        f"  Observations     : "
        f"{observations}"
    )
    print(
        f"  Votes            : "
        f"{votes}"
    )


# ============================================================
# SORT RESULTS
# ============================================================

# Valid first, then highest accuracy
status_priority = {
    "VALID": 0,
    "REVIEW": 1,
    "INVALID": 2,
}

final_results.sort(
    key=lambda item: (
        status_priority.get(
            item["status"],
            3
        ),
        -item["accuracy_score"]
    )
)


# Assign ranks
for rank, result in enumerate(
    final_results,
    start=1
):
    result["rank"] = rank


# ============================================================
# MERGE DUPLICATE FINAL PLATES
# ============================================================

print()
print("MERGING DUPLICATE IMPROVED PLATES")
print("-" * 60)

merged = {}

for result in final_results:

    plate = result["improved_plate"]

    if not plate:
        continue

    if plate not in merged:

        merged[plate] = result.copy()

    else:

        existing = merged[plate]

        existing["observations"] += (
            result["observations"]
        )

        existing["votes"] += (
            result["votes"]
        )

        existing["ocr_readings"] = list(
            dict.fromkeys(
                existing["ocr_readings"]
                + result["ocr_readings"]
            )
        )

        existing["frames"] = list(
            dict.fromkeys(
                existing["frames"]
                + result["frames"]
            )
        )

        existing["accuracy_score"] = max(
            existing["accuracy_score"],
            result["accuracy_score"]
        )

        existing["consistency_score"] = max(
            existing["consistency_score"],
            result["consistency_score"]
        )

        existing["original_confidence"] = max(
            existing["original_confidence"],
            result["original_confidence"]
        )

        # Keep VALID if either result is VALID
        if result["status"] == "VALID":
            existing["status"] = "VALID"

        elif (
            existing["status"] == "INVALID"
            and result["status"] == "REVIEW"
        ):
            existing["status"] = "REVIEW"


final_results = list(
    merged.values()
)


# Re-sort
final_results.sort(
    key=lambda item: (
        status_priority.get(
            item["status"],
            3
        ),
        -item["accuracy_score"]
    )
)


for rank, result in enumerate(
    final_results,
    start=1
):
    result["rank"] = rank


print(
    f"Before merging: "
    f"{len(track_results)}"
)

print(
    f"After merging : "
    f"{len(final_results)}"
)


# ============================================================
# STATISTICS
# ============================================================

valid_results = [
    item for item in final_results
    if item["status"] == "VALID"
]

review_results = [
    item for item in final_results
    if item["status"] == "REVIEW"
]

invalid_results = [
    item for item in final_results
    if item["status"] == "INVALID"
]


# ============================================================
# FINAL REPORT
# ============================================================

video_information = report.get(
    "video_information",
    {}
)

output_report = {

    "project": "AegisVision",

    "module": "ANPR",

    "phase": "5.7",

    "phase_name":
        "OCR Accuracy Improvement",

    "input_phase": "5.5",

    "input_report": str(
        INPUT_REPORT
    ),

    "video_information":
        video_information,

    "processing": {

        "minimum_plate_length":
            MIN_LENGTH,

        "maximum_plate_length":
            MAX_LENGTH,

        "valid_confidence":
            VALID_CONFIDENCE,

        "review_confidence":
            REVIEW_CONFIDENCE,

        "strong_observations":
            STRONG_OBSERVATIONS,

        "ocr_confusion_correction":
            True,

        "multi_reading_consistency":
            True,

        "format_validation":
            True,
    },

    "statistics": {

        "input_tracks":
            len(track_results),

        "final_unique_plates":
            len(final_results),

        "valid_plates":
            len(valid_results),

        "review_plates":
            len(review_results),

        "invalid_plates":
            len(invalid_results),
    },

    "final_plates":
        final_results,

    "output_files": {

        "json":
            str(OUTPUT_JSON),

        "csv":
            str(OUTPUT_CSV),
    },
}


# ============================================================
# SAVE JSON
# ============================================================

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        output_report,
        file,
        indent=4
    )


# ============================================================
# SAVE CSV
# ============================================================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Rank",
        "Track ID",
        "Original Plate",
        "Improved Plate",
        "Status",
        "Original Confidence (%)",
        "Consistency (%)",
        "Format Score (%)",
        "Accuracy Score (%)",
        "Observations",
        "Votes",
        "Frames",
    ])

    for result in final_results:

        writer.writerow([
            result["rank"],
            result["track_id"],
            result["original_plate"],
            result["improved_plate"],
            result["status"],
            result["original_confidence"],
            result["consistency_score"],
            result["format_score"],
            result["accuracy_score"],
            result["observations"],
            result["votes"],
            ",".join(
                str(frame)
                for frame in result["frames"]
            ),
        ])


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("PHASE 5.7 FINAL SUMMARY")
print("=" * 60)

print(
    f"Input tracks        : "
    f"{len(track_results)}"
)

print(
    f"Unique final plates : "
    f"{len(final_results)}"
)

print(
    f"Valid plates        : "
    f"{len(valid_results)}"
)

print(
    f"Review plates       : "
    f"{len(review_results)}"
)

print(
    f"Invalid plates      : "
    f"{len(invalid_results)}"
)


print()
print("FINAL IMPROVED ANPR RESULTS")
print("-" * 60)


for result in final_results:

    print(
        f'{result["rank"]}. '
        f'{result["improved_plate"]} | '
        f'{result["status"]} | '
        f'Accuracy: '
        f'{result["accuracy_score"]:.2f}% | '
        f'Consistency: '
        f'{result["consistency_score"]:.2f}% | '
        f'Observations: '
        f'{result["observations"]} | '
        f'Track: '
        f'{result["track_id"]}'
    )


print()
print("VALID PLATES")
print("-" * 60)

if valid_results:

    for result in valid_results:

        print(
            f'  {result["improved_plate"]}'
        )

else:

    print("  No high-confidence valid plates.")


print()
print("JSON report:")
print(OUTPUT_JSON)

print()
print("CSV report:")
print(OUTPUT_CSV)

print()
print("=" * 60)
print("PHASE 5.7 COMPLETED SUCCESSFULLY")
print("OCR accuracy improvement completed.")
print("=" * 60)