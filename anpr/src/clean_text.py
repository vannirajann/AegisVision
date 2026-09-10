import re

UK_PATTERN = re.compile(r'^[A-Z]{2}\d{2}[A-Z]{3}$')
IN_PATTERN = re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{3,4}$')


def validate_plate_format(plate_text):
    return bool(UK_PATTERN.match(plate_text)) or bool(IN_PATTERN.match(plate_text))


def fix_common_confusions(text):
    candidates = {text}
    candidates.add(text.replace("O", "0"))
    candidates.add(text.replace("0", "O"))
    candidates.add(text.replace("I", "1"))
    candidates.add(text.replace("1", "I"))
    candidates.add(text.replace("S", "5"))
    candidates.add(text.replace("5", "S"))
    candidates.add(text.replace("B", "8"))
    candidates.add(text.replace("8", "B"))

    if len(text) == 7:
        letters1, digits, letters2 = text[:2], text[2:4], text[4:]
        digits_fixed = digits.replace("O", "0").replace("I", "1").replace("S", "5").replace("B", "8")
        letters1_fixed = letters1.replace("0", "O").replace("1", "I").replace("5", "S").replace("8", "B")
        letters2_fixed = letters2.replace("0", "O").replace("1", "I").replace("5", "S").replace("8", "B")
        candidates.add(letters1_fixed + digits_fixed + letters2_fixed)
        candidates.add(letters1 + digits_fixed + letters2)
        candidates.add(letters1_fixed + digits + letters2_fixed)

    return candidates


def clean_ocr_fragments(ocr_results, min_confidence=0.55):
    """Clean + validate OCR fragments.

    Returns (text, valid_format). Confirmed plates validate against a known
    format (directly or via confusion-fix). If the input is too short/noisy
    (<4 chars or below min_confidence) it is rejected as UNKNOWN ("", False).
    A longer reading that happens to match no known format is returned as-is
    with valid=False so the video voter can still weigh an unvalidated but
    long, confident read (the engine gates emission on EVENT_MIN_TEXT_LEN +
    EVENT_MIN_MEAN_CONFIDENCE for those).
    """
    kept = [r for r in ocr_results if r["confidence"] >= min_confidence]

    cleaned_fragments = []
    for r in kept:
        only_alnum = re.sub(r'[^A-Za-z0-9]', '', r["text"]).upper()
        if len(only_alnum) >= 4:
            cleaned_fragments.append({
                "text": only_alnum,
                "confidence": r["confidence"]
            })

    if not cleaned_fragments:
        return "", False

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

    # nothing matched a known format: keep the best (longest, then
    # highest-confidence) reading as an unvalidated candidate
    best_guess = max(cleaned_fragments, key=lambda f: (len(f["text"]), f["confidence"]))
    return best_guess["text"], False


if __name__ == "__main__":
    test_cases = [
        [{"text": "TNO1AK6321", "confidence": 0.65}],
        [{"text": "AP27853676", "confidence": 0.57}],
        [{"text": "GXI5OGJ", "confidence": 0.68}],
        [{"text": "GX150GJ", "confidence": 0.68}],
        [{"text": "HSIU", "confidence": 0.51}],  # should now be rejected: low conf + invalid
    ]
    for case in test_cases:
        result, valid = clean_ocr_fragments(case)
        print(f"Cleaned: '{result}'  Valid format: {valid}")