from fence import RestrictedZone, ZoneEntryDetector

def test_zone_entry():
    zone = RestrictedZone([(150, 300), (450, 300), (450, 550), (150, 550)])
    detector = ZoneEntryDetector(zone)

    track_id = 7

    # Frame 1: object is outside -> no entry event (first time seeing it)
    entered1 = detector.check_entry(track_id, (50, 50))
    print("Frame 1 entered?", entered1)
    assert entered1 == False

    # Frame 2: still outside -> no entry
    entered2 = detector.check_entry(track_id, (60, 60))
    print("Frame 2 entered?", entered2)
    assert entered2 == False

    # Frame 3: object moved INSIDE the zone -> this IS an entry
    entered3 = detector.check_entry(track_id, (300, 400))
    print("Frame 3 entered?", entered3)
    assert entered3 == True

    # Frame 4: still inside -> should NOT fire again (avoid duplicate alerts)
    entered4 = detector.check_entry(track_id, (310, 410))
    print("Frame 4 entered?", entered4)
    assert entered4 == False

    print("✅ Test passed: entry detected once, no duplicate alerts while staying inside")

if __name__ == "__main__":
    test_zone_entry()