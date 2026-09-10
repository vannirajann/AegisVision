from datetime import datetime, timezone


def create_face_event(faces):
    """
    Create a structured event for detected faces.

    Args:
        faces: List of face bounding-box dictionaries.

    Returns:
        Dictionary containing face event information.
    """

    return {
        "event_type": "face_detected",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "face_count": len(faces),
        "faces": faces
    }