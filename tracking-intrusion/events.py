from datetime import datetime, timezone


def create_intrusion_event(object_type, track_id, confidence, severity="high"):
    """
    Builds a single intrusion event in the exact format Person 5's
    backend expects. Called whenever check_crossing() or check_entry()
    returns True.
    """
    event = {
        "event_type": "intrusion",
        "object_type": object_type,
        "track_id": track_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "confidence": confidence,
        "severity": severity
    }
    return event