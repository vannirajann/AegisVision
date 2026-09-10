import json
from datetime import datetime, timezone

def build_anpr_result(plate_text, is_valid, avg_confidence):
    result = {
        "event_type": "anpr",
        "plate_number": plate_text,
        "plate_valid_format": is_valid,
        "confidence": round(avg_confidence, 2),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    return result

if __name__ == "__main__":
    # Quick manual test
    sample = build_anpr_result("TN01AK6321", True, 0.65)
    print(json.dumps(sample, indent=2))