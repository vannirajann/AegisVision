from pipeline import IntrusionPipeline

def make_detection(cx, cy, half_size=25):
    """Helper: builds a detection dict centered around (cx, cy),
    simulating what Person 1's detection module would send."""
    return {
        "class": "person",
        "confidence": 0.95,
        "bbox": {
            "x1": cx - half_size, "y1": cy - half_size * 2,
            "x2": cx + half_size, "y2": cy
        }
    }

def run_demo():
    print("=" * 60)
    print("AEGISVISION - tracking-intrusion module DEMO")
    print("=" * 60)

    zone_points = [(150, 300), (450, 300), (450, 550), (150, 550)]
    print(f"\nRestricted zone defined at: {zone_points}")

    pipeline = IntrusionPipeline(zone_points=zone_points)

    path = [
        (100, 250),  # frame 1: outside
        (120, 270),  # frame 2: outside, approaching
        (140, 290),  # frame 3: outside, right at the edge
        (160, 310),  # frame 4: INSIDE -> should trigger an intrusion event
        (180, 330),  # frame 5: still inside -> no duplicate alert
        (200, 350),  # frame 6: still inside -> no duplicate alert
    ]

    print("\nSimulating a person walking across frames...\n")

    for i, (cx, cy) in enumerate(path, start=1):
        detection = make_detection(cx, cy)
        events = pipeline.process_frame([detection])

        print(f"Frame {i}: person detected at approx ({cx}, {cy})")
        if events:
            for e in events:
                print("  🚨 INTRUSION EVENT GENERATED:")
                for key, value in e.items():
                    print(f"      {key}: {value}")
        else:
            print("  (no event)")

    print("\n" + "=" * 60)
    print("Demo complete.")
    print("=" * 60)

if __name__ == "__main__":
    run_demo()