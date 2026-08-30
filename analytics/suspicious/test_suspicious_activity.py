from suspicious_activity_detector import SuspiciousActivityDetector


print("Starting suspicious activity detection test...")

detector = SuspiciousActivityDetector()


# Test 1: Loitering
print("\nTEST 1: Loitering")

result = detector.analyze(
    track_id=1,
    loitering=True
)

print(result)


# Test 2: Repeated entry
print("\nTEST 2: Repeated Entry")

result = detector.analyze(
    track_id=2,
    repeated_entry=True
)

print(result)


# Test 3: Night movement
print("\nTEST 3: Night Movement")

result = detector.analyze(
    track_id=3,
    night_movement=True
)

print(result)


# Test 4: Multiple suspicious conditions
print("\nTEST 4: Multiple Suspicious Conditions")

result = detector.analyze(
    track_id=4,
    loitering=True,
    repeated_entry=True,
    night_movement=True
)

print(result)


# Test 5: Normal activity
print("\nTEST 5: Normal Activity")

result = detector.analyze(
    track_id=5
)

print(result)


print("\nSuspicious activity test completed")