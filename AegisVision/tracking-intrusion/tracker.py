class CentroidTracker:
    """
    Keeps track of objects across frames using their center points.
    Each object gets a unique ID that stays the same as long as we
    keep seeing it in roughly the same place, frame after frame.
    """

    def __init__(self, max_distance=50, max_missed_frames=10):
        self.next_id = 0
        self.objects = {}       # track_id -> center point (x, y)
        self.missed_frames = {} # track_id -> how many frames since last seen
        self.max_distance = max_distance          # how close counts as "same object"
        self.max_missed_frames = max_missed_frames # how long to keep a lost object before dropping it

    def _distance(self, p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def update(self, new_points):
        """
        new_points: list of center points seen THIS frame, e.g. [(200, 500), (400, 300)]
        Returns: dict of track_id -> center point for objects tracked this frame
        """
        # Case 1: no existing objects yet -> everything is new
        if len(self.objects) == 0:
            for point in new_points:
                self.objects[self.next_id] = point
                self.missed_frames[self.next_id] = 0
                self.next_id += 1
            return dict(self.objects)

        existing_ids = list(self.objects.keys())
        existing_points = list(self.objects.values())

        matched_existing = set()
        matched_new = set()

        # Try to match each new point to the closest existing object
        for new_index, new_point in enumerate(new_points):
            best_id = None
            best_distance = self.max_distance  # must be within this range to count as a match

            for existing_index, existing_id in enumerate(existing_ids):
                if existing_id in matched_existing:
                    continue  # already matched to another new point
                dist = self._distance(new_point, existing_points[existing_index])
                if dist < best_distance:
                    best_distance = dist
                    best_id = existing_id

            if best_id is not None:
                # Same object as before -> keep its ID, update its position
                self.objects[best_id] = new_point
                self.missed_frames[best_id] = 0
                matched_existing.add(best_id)
                matched_new.add(new_index)

        # Any new point that didn't match an existing object -> it's a new object
        for new_index, new_point in enumerate(new_points):
            if new_index not in matched_new:
                self.objects[self.next_id] = new_point
                self.missed_frames[self.next_id] = 0
                self.next_id += 1

        # Any existing object that wasn't matched this frame -> it might have left
        for existing_id in existing_ids:
            if existing_id not in matched_existing:
                self.missed_frames[existing_id] += 1
                if self.missed_frames[existing_id] > self.max_missed_frames:
                    del self.objects[existing_id]
                    del self.missed_frames[existing_id]

        return dict(self.objects)