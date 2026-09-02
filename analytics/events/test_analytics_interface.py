from analytics_interface import AnalyticsInterface


print("Starting analytics backend interface test...")

interface = AnalyticsInterface()


print("\nTEST 1: Face Detection")

event = interface.create_face_event(
    track_id=1,
    face_count=1
)

print(event)


print("\nTEST 2: Night Movement")

event = interface.create_night_movement_event(
    track_id=2,
    brightness=65.9,
    light_status="LOW LIGHT / NIGHT"
)

print(event)


print("\nTEST 3: Loitering")

event = interface.create_loitering_event(
    track_id=3,
    duration_seconds=60
)

print(event)


print("\nTEST 4: Suspicious Activity")

event = interface.create_suspicious_event(
    track_id=4,
    severity="high",
    reasons=[
        "Repeated entry into restricted zone",
        "Night-time movement"
    ]
)

print(event)


print("\nAnalytics backend interface test completed")