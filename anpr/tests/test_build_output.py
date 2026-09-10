import unittest
import sys
import os
import json
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from build_output import build_anpr_result


class TestBuildAnprResult(unittest.TestCase):
    def test_basic_structure(self):
        result = build_anpr_result("TN01AK6321", True, 0.89)
        self.assertIn("event_type", result)
        self.assertIn("plate_number", result)
        self.assertIn("plate_valid_format", result)
        self.assertIn("confidence", result)
        self.assertIn("timestamp", result)

    def test_event_type_is_anpr(self):
        result = build_anpr_result("TN01AK6321", True, 0.89)
        self.assertEqual(result["event_type"], "anpr")

    def test_plate_number_uppercase(self):
        result = build_anpr_result("tn01ak6321", True, 0.89)
        self.assertEqual(result["plate_number"], "tn01ak6321")

    def test_valid_format_flag(self):
        result = build_anpr_result("TN01AK6321", True, 0.89)
        self.assertTrue(result["plate_valid_format"])

        result = build_anpr_result("UNKNOWN", False, 0.3)
        self.assertFalse(result["plate_valid_format"])

    def test_confidence_rounded(self):
        result = build_anpr_result("TN01AK6321", True, 0.8923)
        self.assertEqual(result["confidence"], 0.89)

    def test_timestamp_is_iso_utc(self):
        result = build_anpr_result("TN01AK6321", True, 0.89)
        ts = result["timestamp"]
        self.assertIn("+00:00", ts)
        parsed = datetime.fromisoformat(ts)
        self.assertEqual(parsed.tzinfo, timezone.utc)

    def test_json_serializable(self):
        result = build_anpr_result("TN01AK6321", True, 0.89)
        json_str = json.dumps(result)
        parsed_back = json.loads(json_str)
        self.assertEqual(parsed_back["plate_number"], "TN01AK6321")


if __name__ == '__main__':
    unittest.main()
