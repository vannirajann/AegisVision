from person_tracker import PersonTracker


tracker = PersonTracker()

# Simulate two detected people
detections = [
    (100, 100, 80, 160),
    (300, 120, 90, 170)
]

# Update the tracker
people = tracker.update(detections)

print("People detected:", len(people))

for person in people:
    print(
        "Person ID:",
        person["id"],
        "Bounding box:",
        person["bbox"]
    )

print("Total tracked people:", tracker.get_people_count())