from evidence_manager import EvidenceManager


manager = EvidenceManager()

# Save a test intrusion event
filepath = manager.save_evidence(
    event_type="intrusion",
    person_id=1,
    details={
        "location": "restricted_zone",
        "message": "Person entered restricted area"
    }
)

print("Evidence saved successfully")
print("Evidence file:", filepath)
print("Total evidence files:", manager.get_evidence_count())