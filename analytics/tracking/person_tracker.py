class PersonTracker:
    def __init__(self):
        self.next_id = 1
        self.tracked_people = {}

    def update(self, detections):
        """
        Update tracked people using detected bounding boxes.

        detections should be a list of bounding boxes:
        [(x, y, width, height), ...]
        """

        current_people = []

        for box in detections:
            person_id = self.next_id
            self.next_id += 1

            self.tracked_people[person_id] = box

            current_people.append({
                "id": person_id,
                "bbox": box
            })

        return current_people

    def get_people_count(self):
        """Return the number of currently tracked people."""
        return len(self.tracked_people)

    def reset(self):
        """Clear all tracked people."""
        self.tracked_people.clear()
        self.next_id = 1