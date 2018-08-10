import torch
from margipose.dsntnn import euclidean_losses

from pose3d_utils.coords import ensure_cartesian
from margipose.data.skeleton import absolute_to_root_relative, apply_rigid_alignment, \
    CanonicalSkeletonDesc


def mpjpe(actual, expected, included_joints=None):
    dists = euclidean_losses(actual, expected)
    if included_joints is not None:
        dists = dists.gather(-1, torch.LongTensor(included_joints))
    return dists.mean().item()


def pck(actual, expected, included_joints=None, threshold=150):
    dists = euclidean_losses(actual, expected)
    if included_joints is not None:
        dists = dists.gather(-1, torch.LongTensor(included_joints))
    return (dists < threshold).double().mean().item()


def auc(actual, expected, included_joints=None):
    # This range of thresholds mimics `mpii_compute_3d_pck.m`, which is provided as part of the
    # MPI-INF-3DHP test data release.
    thresholds = torch.linspace(0, 150, 31).tolist()

    pck_values = torch.DoubleTensor(len(thresholds))
    for i, threshold in enumerate(thresholds):
        pck_values[i] = pck(actual, expected, included_joints, threshold=threshold)
    return pck_values.mean().item()


def prepare_for_3d_evaluation(original_skel, norm_pred, dataset, camera_intrinsics, transform_opts,
                              known_depth=False):
    """Process predictions and ground truth into root-relative original skeleton space.

    Args:
        original_skel (torch.Tensor): Ground truth skeleton joint locations in the original
                                      coordinate space.
        norm_pred (torch.Tensor): Normalised predictions for skeleton joints.
        dataset:
        camera_intrinsics:
        transform_opts:
        known_depth (bool): If true, use the ground truth depth of the root joint. If false,
                            use skeleton height of 92cm knee-neck to infer depth.

    Returns:
        Expected and actual skeletons in original coordinate space.
    """
    if known_depth:
        z_ref = original_skel[dataset.skeleton_desc.root_joint_id][2]
        denorm_skel = dataset.denormalise_with_depth(norm_pred, z_ref, camera_intrinsics)
    else:
        denorm_skel = dataset.denormalise_with_skeleton_height(
            norm_pred, camera_intrinsics, transform_opts
        )
    pred_skel = dataset.untransform_skeleton(denorm_skel, transform_opts)
    actual = absolute_to_root_relative(
        dataset.to_canonical_skeleton(ensure_cartesian(pred_skel, d=3)),
        CanonicalSkeletonDesc.root_joint_id
    )
    expected = absolute_to_root_relative(
        dataset.to_canonical_skeleton(ensure_cartesian(original_skel, d=3)),
        CanonicalSkeletonDesc.root_joint_id
    )
    return expected, actual


def gather_3d_metrics(expected, actual, included_joints=None):
    unaligned_mpjpe = mpjpe(actual, expected, included_joints)
    unaligned_pck = pck(actual, expected, included_joints)
    unaligned_auc = auc(actual, expected, included_joints)
    aligned = apply_rigid_alignment(actual, expected)
    aligned_mpjpe = mpjpe(aligned, expected, included_joints)
    aligned_pck = pck(aligned, expected, included_joints)
    aligned_auc = auc(aligned, expected, included_joints)
    return dict(
        mpjpe=unaligned_mpjpe,
        pck=unaligned_pck,
        auc=unaligned_auc,
        aligned_mpjpe=aligned_mpjpe,
        aligned_pck=aligned_pck,
        aligned_auc=aligned_auc,
    )


def calculate_pckh_distance(pred, target, head_length):
    return torch.dist(target, pred) / head_length
