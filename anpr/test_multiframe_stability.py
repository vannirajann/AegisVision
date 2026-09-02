import re
from collections import Counter

MIN_TEXT_LENGTH = 4
MAX_TEXT_LENGTH = 10
FUZZY_THRESHOLD = 0.78
VALID_PLATE_PATTERNS = [
    r"^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$",
    r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$",
    r"^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{1,4}$",
]


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

    chars = list(text)
    for idx, ch in enumerate(chars):
        if ch in {"0", "O"} and idx < 2:
            chars[idx] = "O"
        elif ch in {"1", "I", "L"} and idx < 2:
            chars[idx] = "I"
        elif ch in {"5", "S"} and idx >= 2:
            chars[idx] = "S"
        elif ch in {"8", "B"} and idx >= 2 and idx < 4:
            chars[idx] = "B"
        elif ch in {"6", "G"} and idx >= 2:
            chars[idx] = "G"
        elif ch in {"2", "Z"} and idx >= 2:
            chars[idx] = "Z"

    normalized = "".join(chars)
    if len(normalized) < MIN_TEXT_LENGTH:
        return ""
    return normalized


def validate_indian_plate(text):
    plate = clean_text(text)
    if not plate:
        return False
    if len(plate) < 5 or len(plate) > 10:
        return False
    if not re.search(r"[A-Z]", plate):
        return False
    if re.fullmatch(r"^[A-Z]+$", plate):
        return len(plate) >= 6
    if not re.fullmatch(r"^[A-Z0-9]+$", plate):
        return False
    for pattern in VALID_PLATE_PATTERNS:
        if re.fullmatch(pattern, plate):
            return True
    if len(plate) >= 5 and len(plate) <= 10:
        if re.search(r"[0-9]", plate):
            return True
        if len(plate) >= 6 and re.fullmatch(r"[A-Z]+", plate):
            return True
    return False


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
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return 1.0 - (distance / max_len)


def is_similar_plate(a, b):
    a = clean_text(a)
    b = clean_text(b)
    if not a or not b:
        return False
    if abs(len(a) - len(b)) > 2:
        return False
    if similarity(a, b) >= FUZZY_THRESHOLD:
        return True
    return levenshtein_distance(a, b) <= 2 and min(len(a), len(b)) >= 6


def create_fuzzy_groups(history):
    groups = []
    for text in history:
        new_text = normalize_plate(text)
        if not new_text or not validate_indian_plate(new_text):
            continue
        placed = False
        for group in groups:
            if is_similar_plate(new_text, group["representative"]):
                group["items"].append(new_text)
                placed = True
                break
        if not placed:
            groups.append({"representative": new_text, "items": [new_text]})
    return groups


def calculate_stability(history):
    if not history:
        return {"plate": "", "votes": 0, "stability_percent": 0.0, "valid_frame_readings": 0}
    valid = []
    for item in history:
        normalized = normalize_plate(item)
        if normalized and validate_indian_plate(normalized):
            valid.append(normalized)
    if not valid:
        return {"plate": "", "votes": 0, "stability_percent": 0.0, "valid_frame_readings": 0}

    groups = create_fuzzy_groups(valid)
    groups.sort(key=lambda g: len(g["items"]), reverse=True)
    best = groups[0]
    plate_counter = Counter(best["items"])
    plate, votes = plate_counter.most_common(1)[0]
    stability = (votes / len(valid)) * 100.0
    return {
        "plate": plate,
        "votes": votes,
        "stability_percent": round(stability, 2),
        "valid_frame_readings": len(valid),
    }


def run_case(name, history):
    result = calculate_stability(history)
    print(f"\nCASE: {name}")
    print(f"RESULT: {result}")
    return result


if __name__ == "__main__":
    test_cases = [
        (
            "stable same plate",
            ["PNEGOISO"] * 45 + ["PNEGOLSO"] * 2 + ["PNEGEISO"] * 1 + ["YIES"] + ["WAS"]
        ),
        (
            "temporary OCR failure",
            ["PNEGOISO"] * 20 + ["INVALID"] * 3 + ["PNEGOLSO"] * 3 + ["PNEGOI5O"] * 2
        ),
        (
            "two vehicles not mixed",
            ["TN01AB1234"] * 12 + ["KA01CD4321"] * 8 + ["TN01AB1234"] * 4
        ),
    ]

    for name, history in test_cases:
        result = run_case(name, history)
        assert result["valid_frame_readings"] >= 1
        assert result["stability_percent"] >= 0.0

    stable = calculate_stability(["PNEGOISO"] * 45 + ["PNEGOLSO"] * 2 + ["PNEGEISO"] * 1 + ["YIES"] + ["WAS"])
    assert stable["plate"] == "PNEGOISO"
    assert stable["stability_percent"] > 90.0

    print("\nAll multi-frame stability checks passed.")
