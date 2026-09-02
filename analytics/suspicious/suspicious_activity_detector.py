from datetime import datetime


class SuspiciousActivityDetector:

    def __init__(self):
        self.activity_count = 0

    def analyze(
        self,
        track_id=None,
        loitering=False,
        repeated_entry=False,
        night_movement=False
    ):
        """
        Analyze simple, explainable suspicious-activity rules.
        """

        reasons = []

        # Rule 1: Loitering
        if loitering:
            reasons.append("Person stayed too long in restricted zone")

        # Rule 2: Repeated entry
        if repeated_entry:
            reasons.append("Repeated entry into restricted zone")

        # Rule 3: Night movement
        if night_movement:
            reasons.append("Movement detected during low-light/night conditions")

        # No suspicious activity
        if not reasons:
            return None

        self.activity_count += 1

        # Severity
        if len(reasons) >= 2:
            severity = "high"
        else:
            severity = "medium"

        return {
            "event_type": "suspicious_activity",
            "track_id": track_id,
            "timestamp": datetime.now().isoformat(),
            "severity": severity,
            "reasons": reasons
        }

    def get_activity_count(self):
        return self.activity_count

    def reset(self):
        self.activity_count = 0