import re

def fix_common_confusions(text):
    """Try alternating O<->0 and I<->1 to see if it becomes a valid plate."""
    candidates = {text}

    # Try replacing O with 0 and vice versa
    candidates.add(text.replace("O", "0"))
    candidates.add(text.replace("0", "O"))
    candidates.add(text.replace("I", "1"))
    candidates.add(text.replace("1", "I"))

    return candidates

def validate_plate_format(plate_text):
    # Loosened: allows 0-3 series letters in the middle section
    pattern = r'^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{3,4}$'
    return bool(re.match(pattern, plate_text))

def clean_ocr_fragments(ocr_results, min_confidence=0.25):
    """
    Picks the best single fragment (not merging all), cleans it,
    and tries common OCR-confusion fixes if needed.
    """
    kept = [r for r in ocr_results if r["confidence"] >= min_confidence]

    cleaned_fragments = []
    for r in kept:
        only_alnum = re.sub(r'[^A-Za-z0-9]', '', r["text"]).upper()
        if len(only_alnum) >= 4:  # ignore very short noise fragments
            cleaned_fragments.append({
                "text": only_alnum,
                "confidence": r["confidence"]
            })

    if not cleaned_fragments:
        return "", False

    # Try validating every fragment (with confusion fixes), not just the "best" one.
    # Return the highest-confidence valid fragment if any exist.
    valid_candidates = []
    for f in cleaned_fragments:
        if validate_plate_format(f["text"]):
            valid_candidates.append(f)
        else:
            for candidate in fix_common_confusions(f["text"]):
                if validate_plate_format(candidate):
                    valid_candidates.append({
                        "text": candidate,
                        "confidence": f["confidence"]
                    })
                    break

    if valid_candidates:
        best_valid = max(valid_candidates, key=lambda f: f["confidence"])
        return best_valid["text"], True

    # Nothing validated — return best guess.
    # Prefer fragments that look like real plates: 6-11 chars, starting with 2 letters,
    # then confidence, then proximity to 9.
    def score(f):
        length = len(f["text"])
        if 6 <= length <= 11:
            length_score = length
        else:
            length_score = -10
        prefix_bonus = 2 if re.match(r'^[A-Z]{2}', f["text"]) else 0
        return (f["confidence"] * 3 + prefix_bonus, length_score, -abs(length - 9))

    best = max(cleaned_fragments, key=score)
    return best["text"], False


if __name__ == "__main__":
    test_cases = [
        [{"text": "TNO1AK6321", "confidence": 0.65}],
        [{"text": "AP27853676", "confidence": 0.57}],
        [
            {"text": "BBII 38", "confidence": 0.25},
            {"text": "TN 38", "confidence": 1.00},
            {"text": "DI 3173 N 3173", "confidence": 0.53},
        ],
    ]

    for case in test_cases:
        result, valid = clean_ocr_fragments(case)
        print(f"Cleaned: '{result}'  Valid format: {valid}")