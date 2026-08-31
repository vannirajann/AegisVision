class AnalyticsSummary:
    def __init__(self):
        self.total_people = 0
        self.total_movements = 0
        self.total_events = 0

    def update_people(self, count):
        """Update the number of detected people."""
        self.total_people = count

    def add_movement(self):
        """Record one movement detection."""
        self.total_movements += 1

    def add_event(self):
        """Record one security event."""
        self.total_events += 1

    def get_summary(self):
        """Return the current analytics summary."""
        return {
            "total_people": self.total_people,
            "total_movements": self.total_movements,
            "total_events": self.total_events
        }

    def reset(self):
        """Reset all analytics counters."""
        self.total_people = 0
        self.total_movements = 0
        self.total_events = 0