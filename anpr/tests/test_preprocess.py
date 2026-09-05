import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocess_plate import preprocess_plate, preprocess_plate_gentle


class TestPreprocessPlate(unittest.TestCase):
    def _make_dummy_image(self, width=100, height=40):
        return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    def test_preprocess_returns_grayscale(self):
        img = self._make_dummy_image()
        result = preprocess_plate(img)
        self.assertEqual(len(result.shape), 2)

    def test_preprocess_gentle_returns_grayscale(self):
        img = self._make_dummy_image()
        result = preprocess_plate_gentle(img)
        self.assertEqual(len(result.shape), 2)

    def test_preprocess_resizes_to_target_width(self):
        img = self._make_dummy_image(width=200, height=80)
        result = preprocess_plate(img, target_width=300)
        self.assertEqual(result.shape[1], 300)

    def test_preprocess_gentle_resizes_to_target_width(self):
        img = self._make_dummy_image(width=200, height=80)
        result = preprocess_plate_gentle(img, target_width=300)
        self.assertEqual(result.shape[1], 300)

    def test_preprocess_output_shape_uint8(self):
        img = self._make_dummy_image()
        result = preprocess_plate(img)
        self.assertEqual(result.dtype, np.uint8)

    def test_preprocess_gentle_output_shape_uint8(self):
        img = self._make_dummy_image()
        result = preprocess_plate_gentle(img)
        self.assertEqual(result.dtype, np.uint8)


if __name__ == '__main__':
    unittest.main()
