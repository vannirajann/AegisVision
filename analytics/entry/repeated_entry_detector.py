import time


class RepeatedEntryDetector:

    def __init__(self, max_entries=2):
        self.entry_times = {}
        self.max_entries = max_entries

    def person_entered(self, track_id):
        """
        Record a person's entry into the restricted zone.
        """

        current_time = time.time()

        if track_id not in self.entry_times:
            self.entry_times[track_id] = []

        self.entry_times[track_id].append(current_time)

        entry_count = len(self.entry_times[track_id])

        repeated = entry_count >= self.max_entries

        return {
            "track_id": track_id,
            "entry_count": entry_count,
            "repeated_entry": repeated,
            "timestamp": current_time
        }

    def get_entry_count(self, track_id):
        """
        Return the number of entries for a person.
        """

        return len(self.entry_times.get(track_id, []))

    def reset_person(self, track_id):
        """
        Reset entry history for a person.
        """

        if track_id in self.entry_times:
            del self.entry_times[track_id]