import unittest
import cv2
from find_objects import FindObjects
import numpy as np


class FindObjectsTestCase(unittest.TestCase):
    def setUp(self):
        self.background_img = cv2.imread("a.png")
        self.img = cv2.imread("b.png")
        self.object_finder = FindObjects()
        self.object_finder.crop_region = [20, 180, 1050, 1350]

    def test_find_objects__background_and_normal_image__finds_all_objects(self):
        self.object_finder.background_img = self.background_img
        result = self.object_finder.find_objects(self.img)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 4)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertEqual(result[0].shape[0], 1030)
        result = self.object_finder.find_objects(self.img, crop_to_bbox=True)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 4)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertLess(result[0].shape[0], 500)

    def test_find_objects__edge_cases(self):
        self.object_finder.background_img = self.img
        result = self.object_finder.find_objects(self.img)
        self.assertEqual(result, [])
        self.object_finder.background_img = None
        self.assertRaises(Exception, self.object_finder.find_objects, self.img)