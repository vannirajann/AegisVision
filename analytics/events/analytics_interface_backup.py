try:
    from analytics.events.event_standardizer import EventStandardizer
except ImportError:
    from event_standardizer import EventStandardizer


class AnalyticsInterface:

    def __init__(self):
        self.standardizer = EventStandardizer()

    def create_face_event(self, track_id=None, face_count=1):
        return self.standardizer.create_event(
            event_type="face_detected",
            track_id=track_id,
            severity="low",
            details={
                "face_count": face_count,
                "message": "Face detected"
            }
        )

    def create_night_movement_event(
        self,
        track_id=None,
        brightness=0,
        light_status="LOW LIGHT / NIGHT"
    ):
        return self.standardizer.create_event(
            event_type="night_movement",
            track_id=track_id,
            severity="medium",
            details={
                "brightness": brightness,
                "light_status": light_status,
                "message": "Night-time movement detected"
            }
        )

    def create_loitering_event(
        self,
        track_id,
        duration_seconds
    ):
        return self.standardizer.create_event(
            event_type="loitering",
            track_id=track_id,
            severity="medium",
            details={
                "duration_seconds": duration_seconds,
                "message": "Person remained too long in restricted zone"
            }
        )

    def create_suspicious_event(
        self,
        track_id,
        severity,
        reasons
    ):
        return self.standardizer.create_event(
            event_type="suspicious_activity",
            track_id=track_id,
            severity=severity,
            details={
                "reasons": reasons
            }
        )