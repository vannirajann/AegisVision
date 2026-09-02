from datetime import datetime


class EventGenerator:
    def __init__(self):
        self.events = []

    def create_event(self, event_type, person_id=None, details=None):
        """
        Create a security event.
        """

        event = {
            "event_id": len(self.events) + 1,
            "event_type": event_type,
            "person_id": person_id,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }

        self.events.append(event)

        return event

    def get_events(self):
        """Return all generated events."""
        return self.events

    def get_event_count(self):
        """Return the number of generated events."""
        return len(self.events)

    def clear_events(self):
        """Clear all generated events."""
        self.events.clear()