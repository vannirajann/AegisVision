from pipeline import IntrusionPipeline

def make_detection(cx, cy, half_size=25):
    """Helper: builds a detection dict centered around (cx, cy)."""
    return {
        "class": "person",
        "confidence": 0.95,
        "bbox": {
            "x1": cx - half_size, "y1": cy - half_size * 2,
            "x2": cx + half_size, "y2": cy
        }
    }

def test_pipeline_zone_entry():
    zone_points = [(150, 300), (450, 300), (450, 550), (150, 550)]
    pipeline = IntrusionPipeline(zone_points=zone_points)

    # Person walking gradually toward and into the zone.
    # Each step is ~28px, well under the tracker's max_distance (50px),
    # so the SAME track_id is kept across all frames.
    path = [
        (100, 250),  # frame 1: outside (x < 150)
        (120, 270),  # frame 2: still outside
        (140, 290),  # frame 3: still outside (just before the edge)
        (160, 310),  # frame 4: now inside the zone -> should trigger entry
        (180, 330),  # frame 5: still inside -> should NOT trigger again
    ]

    all_events = []
    for i, (cx, cy) in enumerate(path):
        detections = [make_detection(cx, cy)]
        events = pipeline.process_frame(detections)
        print(f"Frame {i+1} events:", events)
        all_events.extend(events)

    assert len(all_events) == 1, f"Expected exactly 1 event, got {len(all_events)}"
    assert all_events[0]["event_type"] == "intrusion"
    print("✅ Test passed: exactly one intrusion event fired, at the moment of entry")

if __name__ == "__main__":
    test_pipeline_zone_entry()