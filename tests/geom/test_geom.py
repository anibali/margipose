import unittest
from ..common import TestCase

import torch

from margipose.geom import normalise_homogeneous, ensure_cartesian, ensure_homogeneous


class TestCameraIntrinsics(TestCase):
    def test_normalise_homogeneous(self):
        denorm = torch.DoubleTensor([100, 200, 300, 100])
        expected = torch.DoubleTensor([1, 2, 3, 1])
        actual = normalise_homogeneous(denorm)
        self.assertEqual(expected, actual)

    def test_ensure_cartesian(self):
        hom = torch.DoubleTensor([100, 200, 300, 100])
        expected = torch.DoubleTensor([1, 2, 3])
        actual = ensure_cartesian(hom, d=3)
        self.assertEqual(expected, actual)
        self.assertEqual(expected, ensure_cartesian(expected, d=3))

    def test_ensure_homogeneous(self):
        cart = torch.DoubleTensor([1, 2, 3])
        expected = torch.DoubleTensor([1, 2, 3, 1])
        actual = ensure_homogeneous(cart, d=3)
        self.assertEqual(expected, actual)
        self.assertEqual(expected, ensure_homogeneous(expected, d=3))


if __name__ == '__main__':
    unittest.main()
