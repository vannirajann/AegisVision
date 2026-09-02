from events import create_intrusion_event

def test_create_intrusion_event():
    event = create_intrusion_event(
        object_type="person",
        track_id=7,
        confidence=0.95,
        severity="high"
    )

    print("Generated event:", event)

    assert event["event_type"] == "intrusion"
    assert event["object_type"] == "person"
    assert event["track_id"] == 7
    assert event["confidence"] == 0.95
    assert event["severity"] == "high"
    assert "timestamp" in event
    print("✅ Test passed: event built with correct fields")

if __name__ == "__main__":
    test_create_intrusion_event()