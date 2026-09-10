from event_standardizer import EventStandardizer


print("Starting event standardization test...")

standardizer = EventStandardizer()


print("\nTEST 1: Face Detection")

event = standardizer.create_event(
    event_type="face_detected",
    track_id=1,
    severity="low",
    details={
        "face_count": 1,
        "message": "Face detected"
    }
)

print(event)


print("\nTEST 2: Night Movement")

event = standardizer.create_event(
    event_type="night_movement",
    track_id=2,
    severity="medium",
    details={
        "brightness": 65.9,
        "light_status": "LOW LIGHT / NIGHT",
        "message": "Night-time movement detected"
    }
)

print(event)


print("\nTEST 3: Loitering")

event = standardizer.create_event(
    event_type="loitering",
    track_id=3,
    severity="medium",
    details={
        "duration_seconds": 60,
        "message": "Person remained too long in restricted zone"
    }
)

print(event)


print("\nTEST 4: Suspicious Activity")

event = standardizer.create_event(
    event_type="suspicious_activity",
    track_id=4,
    severity="high",
    details={
        "reasons": [
            "Repeated entry into restricted zone",
            "Night-time movement"
        ]
    }
)

print(event)


print("\nTotal standardized events:", standardizer.get_event_count())

print("\nEvent standardization test completed")