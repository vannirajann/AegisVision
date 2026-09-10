from tracker import CentroidTracker

def test_same_object_keeps_id():
    tracker = CentroidTracker()

    # Frame 1: one object appears
    result1 = tracker.update([(200, 500)])
    print("Frame 1:", result1)

    # Frame 2: same object moved slightly -> should KEEP the same ID
    result2 = tracker.update([(205, 502)])
    print("Frame 2:", result2)

    id1 = list(result1.keys())[0]
    id2 = list(result2.keys())[0]
    assert id1 == id2, f"ID changed when it shouldn't have! {id1} vs {id2}"
    print("✅ Test passed: ID stayed the same across frames")

def test_new_object_gets_new_id():
    tracker = CentroidTracker()
    tracker.update([(200, 500)])
    result = tracker.update([(200, 500), (600, 100)])  # a second object appears
    print("Frame with 2 objects:", result)
    assert len(result) == 2, "Should have 2 tracked objects now"
    print("✅ Test passed: new object got a new ID")

if __name__ == "__main__":
    test_same_object_keeps_id()
    test_new_object_gets_new_id()