import time
from fence import RestrictedZone, ZoneEntryDetector

def test_cooldown_prevents_rapid_duplicate_alerts():
    zone = RestrictedZone([(150, 300), (450, 300), (450, 550), (150, 550)])
    detector = ZoneEntryDetector(zone, cooldown_seconds=2)

    track_id = 7

    detector.check_entry(track_id, (50, 50))
    entered1 = detector.check_entry(track_id, (300, 400))
    print("First entry:", entered1)
    assert entered1 == True

    detector.check_entry(track_id, (50, 50))
    entered2 = detector.check_entry(track_id, (300, 400))
    print("Rapid re-entry (should be blocked):", entered2)
    assert entered2 == False

    print("✅ Test passed: cooldown blocked the rapid duplicate alert")

if __name__ == "__main__":
    test_cooldown_prevents_rapid_duplicate_alerts()