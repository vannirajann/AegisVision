class AlertManager:
    def __init__(self):
        self.alerts = []

    def create_alert(self, event_type, person_id=None):
        """
        Create an alert based on the event type.
        """

        if event_type == "intrusion":
            level = "HIGH"
            message = "Intrusion detected in restricted area"

        elif event_type in ("suspicious", "suspicious_activity"):
            level = "MEDIUM"
            message = "Suspicious activity detected"

        else:
            level = "LOW"
            message = "Security event detected"

        alert = {
            "alert_id": len(self.alerts) + 1,
            "level": level,
            "message": message,
            "event_type": event_type,
            "person_id": person_id
        }

        self.alerts.append(alert)

        return alert

    def get_alerts(self):
        """Return all alerts."""
        return self.alerts

    def get_alert_count(self):
        """Return the number of alerts."""
        return len(self.alerts)

    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts.clear()