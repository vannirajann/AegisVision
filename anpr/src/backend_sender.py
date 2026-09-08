import requests

BACKEND_URL = "http://127.0.0.1:8001/events"


def send_anpr_event(event):
    data = {
        "event_type": "anpr",
        "timestamp": event["timestamp"],
        "source": "anpr",
        "severity": "low",
        "data": {
            "plate_number": event["plate_number"],
            "confidence": event["confidence"],
            "plate_valid_format": event["plate_valid_format"],
            "track_id": event.get("track_id"),
            "bbox": event.get("bbox")
        }
    }

    try:
        response = requests.post(
            BACKEND_URL,
            json=data,
            timeout=5
        )

        if response.status_code == 200:
            print("ANPR EVENT SENT SUCCESSFULLY")
            return True

        print("BACKEND ERROR:", response.status_code)
        return False

    except Exception as e:
        print("BACKEND CONNECTION ERROR:", e)
        return False