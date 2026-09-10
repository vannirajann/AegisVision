from event_generator import EventGenerator


generator = EventGenerator()

# Create a test intrusion event
event = generator.create_event(
    event_type="intrusion",
    person_id=1,
    details={
        "location": "restricted_zone",
        "message": "Person entered restricted area"
    }
)

print("Event created successfully")
print("Event ID:", event["event_id"])
print("Event type:", event["event_type"])
print("Person ID:", event["person_id"])
print("Details:", event["details"])
print("Total events:", generator.get_event_count())