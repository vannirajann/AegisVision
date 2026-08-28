from fence import VirtualLine, LineCrossingDetector

def test_line_crossing():
    line = VirtualLine((100, 400), (500, 400))
    detector = LineCrossingDetector(line)

    track_id = 7

    # Frame 1: object is above the line -> no crossing yet (first time seeing it)
    crossed1 = detector.check_crossing(track_id, (300, 200))
    print("Frame 1 crossed?", crossed1)
    assert crossed1 == False

    # Frame 2: still above the line -> still no crossing
    crossed2 = detector.check_crossing(track_id, (310, 210))
    print("Frame 2 crossed?", crossed2)
    assert crossed2 == False

    # Frame 3: object moved BELOW the line -> this IS a crossing
    crossed3 = detector.check_crossing(track_id, (310, 600))
    print("Frame 3 crossed?", crossed3)
    assert crossed3 == True

    print("✅ Test passed: crossing detected only at the moment it happened")

if __name__ == "__main__":
    test_line_crossing()