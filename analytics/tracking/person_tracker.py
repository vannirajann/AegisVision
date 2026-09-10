class PersonTracker:
    def __init__(self, match_distance=80, max_unseen=5):
        self.next_id = 1
        self.tracked_people = {}
        self.centroids = {}
        self.unseen = {}
        self.frame_count = 0
        self.match_distance = match_distance
        self.max_unseen = max_unseen

    def _box_to_centroid(self, box):
        """Convert a detection box to its center point.

        Boxes may be dicts {"x", "y", "width", "height"} (FaceDetector)
        or tuples/lists (x, y, width, height).
        """
        if isinstance(box, dict):
            x = box.get("x", 0)
            y = box.get("y", 0)
            w = box.get("width", 0)
            h = box.get("height", 0)
        else:
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]

        return (int(x + w / 2), int(y + h / 2))

    def update(self, detections):
        """
        Update tracked people using detected bounding boxes.

        detections should be a list of bounding boxes:
        [(x, y, width, height), ...]
        or a list of dicts with x/y/width/height keys.

        The same person keeps their ID across frames while their
        center point stays within match_distance of their last position.
        """

        self.frame_count += 1

        current_people = []
        used_ids = set()

        for box in detections:
            centroid = self._box_to_centroid(box)

            # Find the closest previously tracked person
            best_id = None
            best_dist = self.match_distance

            for person_id, last_centroid in self.centroids.items():

                if person_id in used_ids:
                    continue

                distance = (
                    (last_centroid[0] - centroid[0]) ** 2
                    + (last_centroid[1] - centroid[1]) ** 2
                ) ** 0.5

                if distance < best_dist:
                    best_dist = distance
                    best_id = person_id

            # Reuse the matched ID, otherwise assign a new one
            if best_id is not None:
                person_id = best_id
                used_ids.add(person_id)
                self.unseen[person_id] = 0
            else:
                person_id = self.next_id
                self.next_id += 1
                used_ids.add(person_id)
                self.unseen[person_id] = 0

            self.tracked_people[person_id] = box
            self.centroids[person_id] = centroid

            current_people.append({
                "id": person_id,
                "bbox": box
            })

        # Drop people that have not been seen for too long
        for person_id in list(self.tracked_people.keys()):

            if person_id in used_ids:
                continue

            missed = self.unseen.get(person_id, 0) + 1
            self.unseen[person_id] = missed

            if missed > self.max_unseen:
                del self.tracked_people[person_id]
                del self.centroids[person_id]
                del self.unseen[person_id]

        return current_people

    def get_people_count(self):
        """Return the number of currently tracked people."""
        return len(self.tracked_people)

    def reset(self):
        """Clear all tracked people."""
        self.tracked_people.clear()
        self.centroids.clear()
        self.unseen.clear()
        self.next_id = 1