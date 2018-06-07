import unittest
from .common import TestCase
from .data import UnitTestData as D

import torch

from margipose.data.normalisers import SquareNormaliser, PerspectiveNormaliser, NdcNormaliser
from margipose.data.skeleton import calc_relative_scale, cartesian_to_spherical, \
    absolute_to_parent_relative
from margipose.geom.camera import CameraIntrinsics


class TestSquareNormaliser(TestCase):
    def test_normalise_skeleton(self):
        normaliser = SquareNormaliser()
        skeleton = D.skeleton_in_camera_space()
        root_joint = D.root_joint()
        camera_intrinsics = CameraIntrinsics(D.camera_intrinsic())
        img_w = img_h = 2048

        z_ref = skeleton[root_joint, 2]
        norm_skel = normaliser.normalise_skeleton(skeleton, z_ref, camera_intrinsics, img_h, img_w)

        self.assertLessEqual(norm_skel.max(), 1.0)
        self.assertGreaterEqual(norm_skel.min(), -1.0)

        denormalised_skel = normaliser.denormalise_skeleton(
            norm_skel, z_ref, camera_intrinsics, img_h, img_w)

        self.assertEqual(skeleton, denormalised_skel, 1e-3)

    def test_denormalise_skeleton_autograd(self):
        normaliser = SquareNormaliser()
        skeleton = D.skeleton_in_camera_space()
        root_joint = D.root_joint()
        camera_intrinsics = CameraIntrinsics(D.camera_intrinsic())
        img_w = img_h = 2048
        z_ref = skeleton[root_joint, 2]
        norm_skel = normaliser.normalise_skeleton(skeleton, z_ref, camera_intrinsics, img_h, img_w)

        norm_skel.requires_grad = True
        denorm_skel_var = normaliser.denormalise_skeleton(
            norm_skel, z_ref, camera_intrinsics, img_h, img_w)

        self.assertEqual(denorm_skel_var.data, skeleton)

        denorm_skel_var.sum().backward()
        self.assertIsNotNone(norm_skel.grad)


class TestPerspectiveNormaliser(TestCase):
    def test_denormalise_skeleton(self):
        normaliser = PerspectiveNormaliser()
        img_w = img_h = 2048
        camera_intrinsics = CameraIntrinsics(D.camera_intrinsic())

        coords = torch.Tensor([
            [1, 1, -1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 1],
            [-1, -1, 0, 1],
            [-1, 0, 1, 1],
        ])

        # Expected screen-space coordinates
        expected = torch.Tensor([
            [img_w, img_h],
            [img_w, img_h],
            [img_w, img_h],
            [0, 0],
            [0, img_h / 2],
        ])

        # We don't care about the scale/depth, so z_ref is arbitrary
        z_ref = 100
        denorm_skel = normaliser.denormalise_skeleton(coords, z_ref, camera_intrinsics, img_h, img_w)
        skel_2d = camera_intrinsics.project_cartesian(denorm_skel)

        self.assertEqual(expected, skel_2d[:, :2], 1e-3)

    def test_denormalise_skeleton_autograd(self):
        normaliser = PerspectiveNormaliser()
        skeleton = D.skeleton_in_camera_space()
        root_joint = D.root_joint()
        camera_intrinsics = CameraIntrinsics(D.camera_intrinsic())
        img_w = img_h = 2048
        z_ref = skeleton[root_joint, 2]
        norm_skel = normaliser.normalise_skeleton(skeleton, z_ref, camera_intrinsics, img_h, img_w)

        norm_skel.requires_grad = True
        denorm_skel_var = normaliser.denormalise_skeleton(
            norm_skel, z_ref, camera_intrinsics, img_h, img_w)

        self.assertEqual(denorm_skel_var.data, skeleton)

        denorm_skel_var.sum().backward()
        self.assertIsNotNone(norm_skel.grad)

    def test_normalise_skeleton(self):
        normaliser = PerspectiveNormaliser()
        skeleton = D.skeleton_in_camera_space()
        joint_tree = D.joint_tree()
        root_joint = D.root_joint()
        camera_intrinsics = CameraIntrinsics(D.camera_intrinsic())
        img_w = img_h = 2048

        z_ref = skeleton[root_joint, 2]

        norm_skel = normaliser.normalise_skeleton(skeleton, z_ref, camera_intrinsics, img_h, img_w)

        ref_bone_lengths = cartesian_to_spherical(
            absolute_to_parent_relative(skeleton.narrow(-1, 0, 3), joint_tree)
        )[:, 0]
        def eval_scale(test_skel):
            return calc_relative_scale(test_skel, ref_bone_lengths, joint_tree)

        z_ref = normaliser.infer_depth(norm_skel, eval_scale, camera_intrinsics, img_h, img_w)
        denorm_skel = normaliser.denormalise_skeleton(norm_skel, z_ref, camera_intrinsics, img_h, img_w)

        self.assertEqual(skeleton, denorm_skel, 1e-3)


class TestNdcNormaliser(TestCase):
    def test_normalise_skeleton(self):
        normaliser = NdcNormaliser()
        skeleton = D.skeleton_in_camera_space()
        root_joint = D.root_joint()
        camera_intrinsics = CameraIntrinsics(D.camera_intrinsic())
        width = 2048
        height = 2048
        z_ref = skeleton[root_joint, 2]

        # Normalisation
        norm_skel = normaliser.normalise_skeleton(skeleton, z_ref, camera_intrinsics, height, width)
        # All coords should be in [-1, 1]
        self.assertLessEqual(norm_skel.abs().max(), 1.0)
        # Sign of X and Y coords should be preserved
        self.assertEqual(skeleton[:, :2] < 0, norm_skel[:, :2] < 0)
        # Z-ordering should be preserved
        self.assertEqual(skeleton[:, 2].sort()[1], norm_skel[:, 2].sort()[1])

        # Denormalisation
        recons_skel = normaliser.denormalise_skeleton(norm_skel, z_ref, camera_intrinsics, height, width)
        self.assertEqual(skeleton, recons_skel)


if __name__ == '__main__':
    unittest.main()
