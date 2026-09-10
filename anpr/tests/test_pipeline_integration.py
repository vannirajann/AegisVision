import unittest
import sys
import os
import json
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from anpr_pipeline import save_image_result
    HAS_ANPR = True
except ImportError as e:
    HAS_ANPR = False
    IMPORT_ERROR = str(e)


class TestSaveImageResult(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.original_output = "output"

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @unittest.skipUnless(HAS_ANPR, f"anpr_pipeline dependencies not available: {IMPORT_ERROR if not HAS_ANPR else ''}")
    def test_saves_annotated_image(self):
        import cv2
        image = cv2.imread(os.path.join("test_images", "car.jpg"))
        if image is None:
            self.skipTest("test_images/car.jpg not found")

        detections = [{"bbox": (10, 10, 100, 50), "confidence": 0.9}]
        results = [{
            "event_type": "anpr",
            "plate_number": "TN01AK6321",
            "plate_valid_format": True,
            "confidence": 0.89,
            "timestamp": "2026-09-05T11:00:00+00:00"
        }]

        save_image_result("car.jpg", image, results, detections)

        annotated_path = os.path.join("output", "annotated", "car_result.jpg")
        self.assertTrue(os.path.exists(annotated_path))

    @unittest.skipUnless(HAS_ANPR, f"anpr_pipeline dependencies not available: {IMPORT_ERROR if not HAS_ANPR else ''}")
    def test_saves_results_json(self):
        import cv2
        image = cv2.imread(os.path.join("test_images", "car.jpg"))
        if image is None:
            self.skipTest("test_images/car.jpg not found")

        detections = [{"bbox": (10, 10, 100, 50), "confidence": 0.9}]
        results = [{
            "event_type": "anpr",
            "plate_number": "TN01AK6321",
            "plate_valid_format": True,
            "confidence": 0.89,
            "timestamp": "2026-09-05T11:00:00+00:00"
        }]

        save_image_result("car.jpg", image, results, detections)

        results_path = os.path.join("output", "results", "car_results.json")
        self.assertTrue(os.path.exists(results_path))

        with open(results_path, "r") as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["plate_number"], "TN01AK6321")


if __name__ == '__main__':
    unittest.main()
