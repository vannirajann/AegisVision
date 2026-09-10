from datetime import datetime


class EventStandardizer:

    def __init__(self):
        self.event_count = 0

    def create_event(
        self,
        event_type,
        track_id=None,
        severity="low",
        details=None
    ):
        """
        Create a standardized analytics event.
        """

        self.event_count += 1

        return {
            "event_type": event_type,
            "track_id": track_id,
            "timestamp": datetime.now().isoformat(),
            "severity": severity,
            "details": details or {}
        }

    def get_event_count(self):
        return self.event_count

    def reset(self):
        self.event_count = 0