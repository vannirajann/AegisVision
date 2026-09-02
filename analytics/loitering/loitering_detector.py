import time


class LoiteringDetector:

    def __init__(self, threshold_seconds=10):
        """
        threshold_seconds:
            Maximum allowed time a person can remain
            before being considered loitering.
        """

        self.threshold_seconds = threshold_seconds

        # Stores when each person entered the zone
        self.person_start_times = {}

    def person_entered(self, track_id):
        """
        Start the timer when a person enters the zone.
        """

        if track_id not in self.person_start_times:
            self.person_start_times[track_id] = time.time()

    def person_left(self, track_id):
        """
        Remove the person's timer when they leave.
        """

        if track_id in self.person_start_times:
            del self.person_start_times[track_id]

    def get_duration(self, track_id):
        """
        Return how long the person has remained.
        """

        if track_id not in self.person_start_times:
            return 0

        return time.time() - self.person_start_times[track_id]

    def is_loitering(self, track_id):
        """
        Check whether the person has stayed
        longer than the allowed threshold.
        """

        duration = self.get_duration(track_id)

        return duration >= self.threshold_seconds

    def get_status(self, track_id):
        """
        Return structured loitering information.
        """

        duration = self.get_duration(track_id)

        loitering = duration >= self.threshold_seconds

        return {
            "track_id": track_id,
            "duration_seconds": round(duration, 2),
            "loitering": loitering
        }