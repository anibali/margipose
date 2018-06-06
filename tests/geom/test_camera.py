import unittest
from ..common import TestCase

import torch
import numpy as np

from margipose.geom.camera import CameraIntrinsics


class TestCameraIntrinsics(TestCase):
    def setUp(self):
        self.camera_intrinsics = CameraIntrinsics.from_ccd_params(
            alpha_x=1200, alpha_y=1300, x_0=1000, y_0=500)

    def test_project(self):
        from margipose.geom import homogeneous_to_cartesian

        coords = torch.DoubleTensor([100, 200, 1000, 1])
        actual = homogeneous_to_cartesian(self.camera_intrinsics.project(coords))
        expected = torch.DoubleTensor([1120, 760])
        self.assertEqual(expected, actual)

    def test_back_project(self):
        orig = torch.DoubleTensor([100, 200, 1000, 1])
        proj = torch.DoubleTensor([1120, 760, 1])
        recons = self.camera_intrinsics.back_project(proj)
        # Check that the original point, back-projected point, and camera centre are collinear
        rank = np.linalg.matrix_rank(
            torch.stack([orig[:3], recons[:3], torch.DoubleTensor([0, 0, 0])])
        )
        self.assertEqual(1, rank)


if __name__ == '__main__':
    unittest.main()
