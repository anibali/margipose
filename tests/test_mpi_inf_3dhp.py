import unittest
from .common import TestCase
from .data import UnitTestData as D

from margipose.data.mpi_inf_3dhp import MpiInf3dDataset


class TestMpiInf3dhp(TestCase):
    def test_to_canonical_skeleton(self):
        skel = D.skeleton_in_camera_space()
        expected = skel.new([
            [5.5187, -873.1315, 3685.4100, 1.0000],     # head_top
            [17.8595, -618.7620, 3697.4800, 1.0000],    # neck
            [31.3814, -528.3689, 3570.3000, 1.0000],    # right_shoulder
            [46.0802, -422.7603, 3259.5300, 1.0000],    # right_elbow
            [81.6330, -360.5399, 3018.8300, 1.0000],    # right_wrist
            [-1.5121, -551.9054, 3823.5700, 1.0000],    # left_shoulder
            [-26.8214, -595.2059, 4139.3800, 1.0000],   # left_elbow
            [-24.7079, -610.2892, 4383.5200, 1.0000],   # left_wrist
            [45.3821, -120.6763, 3596.6575, 1.0000],    # right_hip
            [4.2026, 421.7191, 3670.3300, 1.0000],      # right_knee
            [-67.3791, 847.6610, 3718.4000, 1.0000],    # right_ankle
            [-2.3616, -158.1822, 3829.8625, 1.0000],    # left_hip
            [-25.4480, 401.7153, 3800.5900, 1.0000],    # left_knee
            [-117.1699, 836.1045, 3780.2000, 1.0000],   # left_ankle
            [21.5102, -139.4293, 3713.2600, 1.0000],    # pelvis
            [8.0023, -364.9069, 3705.0914, 1.0000],     # spine
            [31.7342, -703.9682, 3696.2600, 1.0000],    # head
        ])

        actual = MpiInf3dDataset._mpi_inf_3dhp_to_canonical_skeleton(skel)
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
