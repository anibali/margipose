"""Utility functions for manipulating joint coordinates."""

import numpy as np
import torch
from pose3d_utils.coords import ensure_cartesian, ensure_homogeneous, homogeneous_to_cartesian
from scipy.spatial import procrustes

# 14-joint skeleton used in the VNect paper for some evaluation
VNect_Common_Skeleton = [
    # 0-4
    'head_top', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist',
    # 5-9
    'left_shoulder', 'left_elbow', 'left_wrist', 'right_hip', 'right_knee',
    # 10-13
    'right_ankle', 'left_hip', 'left_knee', 'left_ankle'
]


class SkeletonDesc:
    def __init__(self, joint_names, joint_tree, hflip_indices):
        self.joint_names = joint_names
        self.joint_tree = joint_tree
        self.hflip_indices = hflip_indices

    @property
    def n_joints(self):
        return len(self.joint_names)

    @property
    def canonical(self):
        # NOTE: A more thorough check would be nice
        return self.n_joints == CanonicalSkeletonDesc.n_joints \
               and self.joint_names == CanonicalSkeletonDesc.joint_names

    @property
    def root_joint_id(self):
        return self.joint_names.index('pelvis')

    def to_dict(self):
        return {
            'joint_names': self.joint_names,
            'joint_tree': self.joint_tree,
            'hflip_indices': self.hflip_indices,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d['joint_names'], d['joint_tree'], d['hflip_indices'])


CanonicalSkeletonDesc = SkeletonDesc(
    joint_names=[
        # 0-4
        'head_top', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist',
        # 5-9
        'left_shoulder', 'left_elbow', 'left_wrist', 'right_hip', 'right_knee',
        # 10-14
        'right_ankle', 'left_hip', 'left_knee', 'left_ankle', 'pelvis',
        # 15-16
        'spine', 'head',
    ],
    joint_tree=[
        1, 15, 1, 2, 3,
        1, 5, 6, 14, 8,
        9, 14, 11, 12, 14,
        14, 1
    ],
    hflip_indices=[
        0, 1, 5, 6, 7,
        2, 3, 4, 11, 12,
        13, 8, 9, 10, 14,
        15, 16
    ]
)


def absolute_to_parent_relative(joints, joint_tree):
    parent_indices = torch.LongTensor(joint_tree).unsqueeze(-1).expand_as(joints)
    parents = joints.gather(-2, parent_indices)
    return joints - parents


def parent_relative_to_absolute(relative, joint_tree):
    absolute = relative.clone().zero_()
    for j, parent in enumerate(joint_tree):
        a, b = j, parent
        while a != b:
            absolute.narrow(-2, j, 1).add_(relative.narrow(-2, a, 1))
            a, b = b, joint_tree[b]
    return absolute


def absolute_to_root_relative(joints, root_index):
    root = joints.narrow(-2, root_index, 1)
    return joints - root


def cartesian_to_spherical(cartesian):
    x, y, z = [t.squeeze(-1) for t in cartesian.split(1, -1)]
    r = (cartesian ** 2).sum(-1, keepdim=False).sqrt()
    theta = (z / r).acos()
    phi = torch.atan2(y, x)
    return torch.stack([r, theta, phi], -1)


def spherical_to_cartesian(spherical):
    r, theta, phi = [t.squeeze(-1) for t in spherical.split(1, -1)]
    sin_theta = theta.sin()
    x = r * sin_theta * phi.cos()
    y = r * sin_theta * phi.sin()
    z = r * theta.cos()
    return torch.stack([x, y, z], -1)


def calc_relative_scale(skeleton, ref_bone_lengths, joint_tree) -> (float, float):
    """Calculate the factor by which the reference is larger than the query skeleton.

    Args:
        skeleton (torch.DoubleTensor): The query skeleton.
        ref_bone_lengths (torch.DoubleTensor): The reference skeleton bone lengths.
        joint_tree (list of int):

    Returns:
        The average scale factor.
    """

    bone_lengths = cartesian_to_spherical(
        absolute_to_parent_relative(ensure_cartesian(skeleton, d=3), joint_tree)
    )[:, 0]

    non_zero = bone_lengths.gt(1e-6)
    if non_zero.sum() == 0: return 0
    ratio = (ref_bone_lengths / bone_lengths).masked_select(non_zero)

    return ratio.median().item()


def bone_path_length(sph_rel_joints, joint_a, joint_b, joint_tree):
    parent_a = joint_tree[joint_a]
    parent_b = joint_tree[joint_b]

    if parent_a != joint_a:
        return sph_rel_joints[joint_a, 0] + \
               bone_path_length(sph_rel_joints, parent_a, joint_b, joint_tree)
    elif parent_b != joint_b:
        return sph_rel_joints[joint_b, 0] + \
               bone_path_length(sph_rel_joints, joint_a, parent_b, joint_tree)
    else:
        return 0


def calculate_knee_neck_height(skel, joint_names):
    """Calculate skeleton height from left knee to neck via the spine joint.

    This function is based on a code snippet provided courtesy of Dushyant Mehta.

    Args:
        skel (torch.Tensor): The skeleton.
        joint_names (list): List of joint names for the skeleton.

    Returns:
        The knee-neck height of the skeleton.
    """

    left_knee = joint_names.index('left_knee')
    left_hip = joint_names.index('left_hip')
    spine = joint_names.index('spine')
    pelvis = joint_names.index('pelvis')
    neck = joint_names.index('neck')

    skel = ensure_cartesian(skel, d=3)

    return sum([
        (skel[left_knee] - skel[left_hip]).norm(2).item(),
        (skel[spine] - skel[pelvis]).norm(2).item(),
        (skel[neck] - skel[spine]).norm(2).item(),
    ])


def apply_rigid_alignment(skel, ref_skel):
    """Align a skeleton to a reference skeleton using only rigid transformations."""

    skel = np.array(skel)
    ref_skel = np.array(ref_skel)

    _, mtx2, _ = procrustes(ref_skel, skel)

    # De-normalize the aligned joints
    mean = np.mean(ref_skel, 0)
    stddev = np.linalg.norm(ref_skel - mean)
    aligned = (mtx2 * stddev) + mean

    return torch.from_numpy(aligned)


def make_eval_scale_skeleton_height(skel_desc, untransform):
    target_sum = 920  # Desired skeleton height in mm (from knee to neck)
    joint_names = skel_desc.joint_names
    def eval_scale(test_skel):
        skel = untransform(test_skel)
        return target_sum / (calculate_knee_neck_height(skel, joint_names) + 1e-12)
    return eval_scale


def make_eval_scale_bone_lengths(skel_desc, untransform, ref_skel):
    joint_tree = skel_desc.joint_tree
    ref_bone_lengths = cartesian_to_spherical(
        absolute_to_parent_relative(ensure_cartesian(ref_skel, d=3), joint_tree)
    )[:, 0]
    def eval_scale(test_skel):
        skel = untransform(test_skel)
        return calc_relative_scale(skel, ref_bone_lengths, joint_tree)
    return eval_scale


def canonicalise_orientation(skel_desc, skel):
    """Rotate the skeleton into a canonical orientation.

    This is achieved by aligning the plane formed by the left shoulder, right shoulder,
    and pelvis joints with the XY plane. The root joint is positioned at the origin.
    The direction from the pelvis to the midpoint of the soldiers is aligned
    with the negative Y direction. "Forwards" for the skeleton corresponds to
    the negative Z direction.

    Args:
        skel_desc (SkeletonDesc): The skeleton description
        skel (torch.Tensor): The skeleton

    Returns:
        The re-oriented skeleton
    """
    skel = ensure_homogeneous(skel, d=3)

    cart_skel = homogeneous_to_cartesian(skel)
    cart_skel = cart_skel - cart_skel[skel_desc.root_joint_id]
    rshoulder = cart_skel[skel_desc.joint_names.index('right_shoulder')]
    lshoulder = cart_skel[skel_desc.joint_names.index('left_shoulder')]
    pelvis = cart_skel[skel_desc.joint_names.index('pelvis')]

    v1 = rshoulder - pelvis
    v2 = lshoulder - pelvis
    forward = torch.cross(v1, v2)
    forward = forward / forward.norm(2)

    up = 0.5 * (v1 + v2)
    up = up / up.norm(2)

    right = torch.cross(forward, up)
    right = right / right.norm(2)

    up = torch.cross(forward, right)

    look_at = skel.new([
        [right[0], up[0], forward[0], 0],
        [right[1], up[1], forward[1], 0],
        [right[2], up[2], forward[2], 0],
        [0, 0, 0, 1],
    ])

    return torch.matmul(ensure_homogeneous(cart_skel, d=3), look_at)
