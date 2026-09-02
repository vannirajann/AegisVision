import re
import json
from collections import Counter


# ============================================================
# AEGISVISION - ANPR
# PHASE 6.6 - STEP 3
# OCR TEXT CLEANING + VALIDATION
# ============================================================

print("=" * 70)
print("AEGISVISION - ANPR")
print("PHASE 6.6 - STEP 3")
print("OCR TEXT CLEANING + VALIDATION")
print("=" * 70)


# ============================================================
# INPUT OCR RESULTS
# ============================================================

INPUT_JSON = (
    "output/phase6_6/ocr_tests_v2/"
    "ocr_test_results.json"
)

OUTPUT_JSON = (
    "output/phase6_6/"
    "validated_plate_result.json"
)


# ============================================================
# LOAD OCR RESULTS
# ============================================================

try:

    with open(
        INPUT_JSON,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

except Exception as e:

    print("\nERROR loading OCR results:")
    print(e)

    raise SystemExit


results = data.get(
    "results",
    []
)


# ============================================================
# EXTRACT OCR TEXT
# ============================================================

raw_texts = []

for result in results:

    text = result.get(
        "text",
        ""
    )

    if text:

        raw_texts.append(
            text.upper()
        )


print("\nRaw OCR readings:")

for text in raw_texts:

    print(
        f"  {text}"
    )


# ============================================================
# BASIC CLEANING
# ============================================================

def clean_text(text):

    text = text.upper()

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text


cleaned_texts = [
    clean_text(text)
    for text in raw_texts
]


# ============================================================
# REMOVE VERY SHORT RESULTS
# ============================================================

valid_length_texts = [
    text
    for text in cleaned_texts
    if len(text) >= 5
]


print("\nAfter basic cleaning:")

for text in valid_length_texts:

    print(
        f"  {text}"
    )


# ============================================================
# VOTING
# ============================================================

if not valid_length_texts:

    print("\nNo usable OCR text.")

    raise SystemExit


counter = Counter(
    valid_length_texts
)


print("\nOCR frequency:")

for text, count in counter.most_common():

    print(
        f"  {text:15} -> {count} vote(s)"
    )


raw_best = counter.most_common(1)[0][0]

raw_votes = counter.most_common(1)[0][1]


# ============================================================
# CHARACTER CONFUSION CORRECTION
# ============================================================

def normalize_ocr(text):

    """
    Conservative OCR correction.

    Only correct common OCR confusions when
    the character pattern strongly suggests it.
    """

    chars = list(text)

    # --------------------------------------------------------
    # For a 7-character test plate:
    #
    # Positions 0-3 -> letters
    # Positions 4-6 -> numbers
    #
    # Example:
    # NEGO150
    # --------------------------------------------------------

    if len(chars) == 7:

        # Expected letters
        letter_map = {
            "0": "O",
            "1": "I",
            "5": "S",
            "8": "B"
        }

        for i in range(0, 4):

            if chars[i] in letter_map:

                chars[i] = letter_map[
                    chars[i]
                ]

        # Expected numbers
        number_map = {
            "O": "0",
            "D": "0",
            "I": "1",
            "L": "1",
            "Z": "2",
            "S": "5",
            "G": "6",
            "T": "7",
            "B": "8"
        }

        for i in range(4, 7):

            if chars[i] in number_map:

                chars[i] = number_map[
                    chars[i]
                ]

    return "".join(chars)


# ============================================================
# NORMALIZE EACH OCR RESULT
# ============================================================

normalized_texts = []

print("\nNormalized OCR readings:")

for text in valid_length_texts:

    normalized = normalize_ocr(
        text
    )

    normalized_texts.append(
        normalized
    )

    print(
        f"  {text:15} -> {normalized}"
    )


# ============================================================
# FINAL VOTE
# ============================================================

normalized_counter = Counter(
    normalized_texts
)


print("\nNormalized voting:")

for text, count in (
    normalized_counter.most_common()
):

    print(
        f"  {text:15} -> {count} vote(s)"
    )


final_plate, final_votes = (
    normalized_counter.most_common(1)[0]
)


# ============================================================
# VALIDATION
# ============================================================

def validate_plate(text):

    # General ANPR validation:
    # 5 to 12 alphanumeric characters

    if not 5 <= len(text) <= 12:

        return False, "Invalid length"

    if not re.fullmatch(
        r"[A-Z0-9]+",
        text
    ):

        return False, "Invalid characters"

    return True, "Valid ANPR text"


is_valid, validation_message = (
    validate_plate(final_plate)
)


# ============================================================
# CONFIDENCE
# ============================================================

total_votes = len(
    normalized_texts
)

if total_votes > 0:

    stability = (
        final_votes /
        total_votes
    )

else:

    stability = 0.0


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION RESULT")
print("=" * 70)

print(
    f"Raw best OCR       : {raw_best}"
)

print(
    f"Raw votes          : {raw_votes}"
)

print(
    f"Final plate        : {final_plate}"
)

print(
    f"Final votes        : {final_votes}"
)

print(
    f"Total OCR readings : {total_votes}"
)

print(
    f"Stability          : "
    f"{stability * 100:.2f}%"
)

print(
    f"Validation         : "
    f"{'PASS' if is_valid else 'FAIL'}"
)

print(
    f"Message            : "
    f"{validation_message}"
)


# ============================================================
# SAVE RESULT
# ============================================================

final_result = {

    "raw_best_ocr": raw_best,

    "raw_votes": raw_votes,

    "final_plate": final_plate,

    "final_votes": final_votes,

    "total_ocr_readings": total_votes,

    "stability": stability,

    "validation": is_valid,

    "validation_message": validation_message,

    "all_raw_readings": raw_texts,

    "all_normalized_readings":
        normalized_texts

}


with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        final_result,
        f,
        indent=4
    )


print("\nResult saved:")
print(OUTPUT_JSON)

print("\n" + "=" * 70)
print("PHASE 6.6 STEP 3 COMPLETE")
print("=" * 70)