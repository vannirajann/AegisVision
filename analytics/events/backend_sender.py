import requests


BACKEND_URL = "http://127.0.0.1:8001/events"


def send_event(event):
    data = {
        "event_type": event["event_type"],
        "timestamp": event["timestamp"],
        "source": "analytics",
        "severity": event["severity"],
        "data": {
            "track_id": event["track_id"],
            **event["details"]
        }
    }

    try:
        response = requests.post(
            BACKEND_URL,
            json=data,
            timeout=5
        )

        if response.status_code == 200:
            print("ANALYTICS EVENT SENT SUCCESSFULLY")
            return True

        print("BACKEND ERROR:", response.status_code)
        return False

    except Exception as e:
        print("BACKEND CONNECTION ERROR:", e)
        return False