import numpy as np
import torch
from torch.testing import assert_allclose

from margipose.data.skeleton import spherical_to_cartesian, cartesian_to_spherical, \
    absolute_to_root_relative, absolute_to_parent_relative, parent_relative_to_absolute, \
    limb_dirs_to_skeleton, CanonicalSkeletonDesc, canonicalise_orientation


def test_spherical_to_cartesian():
    spherical = torch.Tensor([[4 * np.sqrt(3), np.deg2rad(30), np.deg2rad(60)]])
    expected = torch.Tensor([[np.sqrt(3), 3, 6]])
    actual = spherical_to_cartesian(spherical)
    assert_allclose(actual, expected)


def test_cartesian_to_spherical():
    cartesian = torch.Tensor([[np.sqrt(3), 3, 6]])
    expected = torch.Tensor([[4 * np.sqrt(3), np.deg2rad(30), np.deg2rad(60)]])
    actual = cartesian_to_spherical(cartesian)
    assert_allclose(actual, expected)


def test_absolute_to_root_relative():
    joints = torch.Tensor([
        [1, 1, 1],
        [1, 2, 1],
        [1, 2, 2],
    ])
    root_joint = 0
    expected = torch.Tensor([
        [0, 0, 0],
        [0, 1, 0],
        [0, 1, 1],
    ])
    actual = absolute_to_root_relative(joints, root_joint)
    assert_allclose(actual, expected)


def test_absolute_to_parent_relative():
    joints = torch.Tensor([
        [1, 1, 1],
        [1, 2, 1],
        [1, 2, 2],
    ])
    joint_tree = [0, 0, 1]
    expected = torch.Tensor([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    actual = absolute_to_parent_relative(joints, joint_tree)
    assert_allclose(actual, expected)


def test_parent_relative_to_absolute():
    relative = torch.Tensor([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    joint_tree = [0, 0, 1]
    expected = torch.Tensor([
        [0, 0, 0],
        [0, 1, 0],
        [0, 1, 1],
    ])
    actual = parent_relative_to_absolute(relative, joint_tree)
    assert_allclose(actual, expected)


def test_limb_dirs_to_skeleton(skeleton_canonical_univ):
    root_joint_id = CanonicalSkeletonDesc.root_joint_id
    skeleton = absolute_to_root_relative(skeleton_canonical_univ, root_joint_id)
    joint_tree = CanonicalSkeletonDesc.joint_tree
    limbs = [(j, parent) for j, parent in enumerate(joint_tree) if j != parent]
    limb_dirs = absolute_to_parent_relative(skeleton, joint_tree)\
        .index_select(-2, torch.LongTensor([j for j, _ in limbs]))
    limb_dirs.div_(-limb_dirs.norm(2, -1, keepdim=True))
    actual = limb_dirs_to_skeleton(limb_dirs, limbs)
    assert_allclose(actual, skeleton.narrow(-1, 0, 3), rtol=0, atol=200)


def test_canonicalise_orientation(skeleton_canonical_univ):
    skel_desc = CanonicalSkeletonDesc

    new_skel = canonicalise_orientation(skel_desc, skeleton_canonical_univ)

    pelvis = new_skel[skel_desc.joint_names.index('pelvis')]
    lshoulder = new_skel[skel_desc.joint_names.index('left_shoulder')]
    rshoulder = new_skel[skel_desc.joint_names.index('right_shoulder')]

    assert_allclose(pelvis, torch.Tensor([0, 0, 0, 1]))
    assert_allclose(lshoulder[2].item(), 0)
    assert_allclose(rshoulder[2].item(), 0)
