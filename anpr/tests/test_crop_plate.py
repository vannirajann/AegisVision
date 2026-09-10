import unittest
import sys
import os
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from crop_plate import crop_plate, save_crop


class TestCropPlate(unittest.TestCase):
    def _make_image(self, width=400, height=300):
        return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    def test_crop_within_bounds(self):
        img = self._make_image()
        bbox = (50, 50, 150, 100)
        cropped = crop_plate(img, bbox, padding=0)
        self.assertEqual(cropped.shape, (50, 100, 3))

    def test_crop_with_padding(self):
        img = self._make_image()
        bbox = (5, 5, 15, 15)
        cropped = crop_plate(img, bbox, padding=10)
        self.assertEqual(cropped.shape, (25, 25, 3))

    def test_crop_does_not_exceed_image(self):
        img = self._make_image(width=100, height=100)
        bbox = (0, 0, 100, 100)
        cropped = crop_plate(img, bbox, padding=10)
        self.assertEqual(cropped.shape, (100, 100, 3))

    def test_crop_at_top_left_corner(self):
        img = self._make_image()
        bbox = (0, 0, 50, 50)
        cropped = crop_plate(img, bbox, padding=0)
        self.assertEqual(cropped.shape, (50, 50, 3))

    def test_save_crop_creates_file(self):
        img = self._make_image()
        output_dir = os.path.join("output_crops", "test_temp")
        os.makedirs(output_dir, exist_ok=True)
        path = save_crop(img, "temp_crop.jpg", output_dir=output_dir)
        self.assertTrue(os.path.exists(path))
        os.remove(path)
        os.rmdir(output_dir)


if __name__ == '__main__':
    unittest.main()
