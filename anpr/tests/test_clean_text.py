import unittest
import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from clean_text import (
    clean_ocr_fragments,
    validate_plate_format,
    fix_common_confusions
)


class TestValidatePlateFormat(unittest.TestCase):
    def test_valid_standard_indian(self):
        self.assertTrue(validate_plate_format("TN01AK6321"))
        self.assertTrue(validate_plate_format("DL2C1234"))
        self.assertTrue(validate_plate_format("MH12AB1234"))
        self.assertTrue(validate_plate_format("KA03P4567"))

    def test_invalid_too_short(self):
        self.assertFalse(validate_plate_format("TN01"))
        self.assertFalse(validate_plate_format("AB123"))

    def test_invalid_too_long(self):
        self.assertFalse(validate_plate_format("TN01AK63210"))
        self.assertFalse(validate_plate_format("ABCD12345"))

    def test_invalid_pattern(self):
        self.assertFalse(validate_plate_format("1234ABCD"))
        self.assertFalse(validate_plate_format("TN01A"))
        self.assertFalse(validate_plate_format("TN01AK63"))

    def test_valid_zero_series_letters(self):
        self.assertTrue(validate_plate_format("TN01A6321"))
        self.assertTrue(validate_plate_format("DL2C1234"))


class TestFixCommonConfusions(unittest.TestCase):
    def test_returns_original_plus_variants(self):
        candidates = fix_common_confusions("TNO1AK6321")
        self.assertIn("TNO1AK6321", candidates)
        self.assertIn("TN01AK6321", candidates)
        self.assertIn("TN01AK6321", candidates)

    def test_zero_one_swaps(self):
        candidates = fix_common_confusions("O1")
        self.assertIn("01", candidates)
        self.assertIn("O1", candidates)


class TestCleanOcrFragments(unittest.TestCase):
    def test_empty_input(self):
        text, valid = clean_ocr_fragments([])
        self.assertEqual(text, "")
        self.assertFalse(valid)

    def test_filters_low_confidence(self):
        fragments = [
            {"text": "TN01", "confidence": 0.3},
            {"text": "TN01AK6321", "confidence": 0.9},
        ]
        text, valid = clean_ocr_fragments(fragments, min_confidence=0.4)
        self.assertEqual(text, "TN01AK6321")
        self.assertTrue(valid)

    def test_ignores_short_noise(self):
        fragments = [
            {"text": "AB", "confidence": 0.9},
            {"text": "TN01AK6321", "confidence": 0.8},
        ]
        text, valid = clean_ocr_fragments(fragments)
        self.assertEqual(text, "TN01AK6321")

    def test_prefers_plate_length(self):
        fragments = [
            {"text": "TN01AK6321", "confidence": 0.7},
            {"text": "TN01AK63210XYZ", "confidence": 0.9},
        ]
        text, valid = clean_ocr_fragments(fragments)
        self.assertEqual(text, "TN01AK6321")

    def test_strips_special_chars(self):
        fragments = [
            {"text": "TN-01-AK-6321", "confidence": 0.9},
        ]
        text, valid = clean_ocr_fragments(fragments)
        self.assertEqual(text, "TN01AK6321")

    def test_ocr_confusion_fix(self):
        fragments = [
            {"text": "TNO1AK6321", "confidence": 0.9},
        ]
        text, valid = clean_ocr_fragments(fragments)
        self.assertEqual(text, "TN01AK6321")
        self.assertTrue(valid)

    def test_returns_invalid_when_no_match(self):
        fragments = [
            {"text": "HELLOWORLD", "confidence": 0.9},
        ]
        text, valid = clean_ocr_fragments(fragments)
        self.assertEqual(text, "HELLOWORLD")
        self.assertFalse(valid)


if __name__ == '__main__':
    unittest.main()
