from repeated_entry_detector import RepeatedEntryDetector
import time


print("Starting repeated-entry detection test...")

detector = RepeatedEntryDetector(max_entries=2)

track_id = 1

print("\nPerson enters restricted zone")

result = detector.person_entered(track_id)

print(
    f"Track ID: {result['track_id']} | "
    f"Entry Count: {result['entry_count']} | "
    f"Repeated Entry: {result['repeated_entry']}"
)

time.sleep(2)

print("\nPerson leaves the zone")

time.sleep(1)

print("\nPerson enters restricted zone again")

result = detector.person_entered(track_id)

print(
    f"Track ID: {result['track_id']} | "
    f"Entry Count: {result['entry_count']} | "
    f"Repeated Entry: {result['repeated_entry']}"
)

if result["repeated_entry"]:
    print("\nREPEATED ENTRY DETECTED")

print("\nRepeated-entry test completed")