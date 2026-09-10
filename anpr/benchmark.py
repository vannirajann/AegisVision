"""Accuracy benchmark for the ANPR engine against ground_truth.json.

Reports exact-match rate, fuzzy-match rate, and valid-format rate for the
image test set. Works against whatever plates the system actually detects —
nothing is hardcoded.

Usage:
    python benchmark.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from anpr_engine import ANPREngine, similarity


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    gt_path = os.path.join(root, "ground_truth.json")
    with open(gt_path, "r") as f:
        ground_truth = json.load(f)

    engine = ANPREngine()

    image_dir = os.path.join(root, "test_images")
    samples = []
    for filename, expected in ground_truth.items():
        path = os.path.join(image_dir, filename)
        # ground truth file lists car.jpg/bike.jpg/bus.jpg; the folder holds .jpeg files
        if not os.path.exists(path):
            alt = os.path.join(image_dir, filename.replace(".jpg", ".jpeg"))
            if os.path.exists(alt):
                path = alt
        if os.path.exists(path):
            samples.append((filename, path, expected))

    n_exact = n_fuzzy = n_valid = n_total_expected = n_total_plates = 0
    for name, path, expected in samples:
        results = engine.detect_image(path, save_output=False)
        detected = [r["plate_text"].upper() for r in results]
        valid = [r["plate_valid_format"] for r in results]

        n_total_plates += len(detected)
        n_valid += sum(1 for v in valid if v)

        print(f"\n=== {name} ===")
        if not detected:
            print("  no plates detected")
        for r in results:
            print(f"  → '{r['plate_text']}'  valid={r['plate_valid_format']}  "
                  f"ocr_conf={r['ocr_confidence']:.2f}  veh={r['vehicle_type']}")

        for expected_plate in map(str.upper, expected):
            n_total_expected += 1
            if expected_plate in detected:
                n_exact += 1
                n_fuzzy += 1
                print(f"  ✓ exact match: {expected_plate}")
            else:
                best = max((similarity(d, expected_plate) for d in detected), default=0.0)
                if best >= 0.75:
                    n_fuzzy += 1
                    print(f"  ~ fuzzy match: expected {expected_plate}, "
                          f"best similarity {best:.2f}")
                else:
                    print(f"  ✗ miss: expected {expected_plate}, "
                          f"best similarity {best:.2f}")

    print("\n================ SUMMARY ================")
    print(f"Samples: {len(samples)}")
    if n_total_expected:
        print(f"Exact-match rate: {n_exact}/{n_total_expected} "
              f"= {100 * n_exact / n_total_expected:.1f}%")
        print(f"Fuzzy-match rate: {n_fuzzy}/{n_total_expected} "
              f"= {100 * n_fuzzy / n_total_expected:.1f}%")
    if n_total_plates:
        print(f"Plates detected: {n_total_plates} "
              f"(valid-format: {n_valid} = {100 * n_valid / n_total_plates:.1f}%)")


if __name__ == "__main__":
    main()