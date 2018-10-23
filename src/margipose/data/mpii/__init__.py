"""
Data loader for the MPII 2D human pose dataset.

Dataset home page: http://human-pose.mpi-inf.mpg.de/
"""

import numpy as np
import torch
import torch.nn.functional
from pose3d_utils.camera import CameraIntrinsics
from torchdata.mpii import MPII_Joint_Names, MPII_Joint_Parents, MPII_Joint_Horizontal_Flips, \
    MpiiData

from margipose.data import PoseDataset
from margipose.data.skeleton import SkeletonDesc, CanonicalSkeletonDesc
from margipose.data_specs import DataSpecs, ImageSpecs, JointsSpecs


MpiiSkeletonDesc = SkeletonDesc(joint_names=MPII_Joint_Names, joint_tree=MPII_Joint_Parents,
                                hflip_indices=MPII_Joint_Horizontal_Flips)


class MpiiDataset(PoseDataset):
    '''Create a Dataset object for loading MPII Human Pose data.

    Args:
        data_dir (str): path to the directory containing `images/` and `mpii_annot_*.h5`
        data_specs (DataSpecs):
        subset (str): subset of the data to load ("train", "val", "trainval", or "test")
        use_aug (bool): set to `True` to enable random data augmentation
        max_length (int): cap the number of examples in the dataset
    '''

    def __init__(self, data_dir, data_specs=None, subset='train', use_aug=False, max_length=None):
        if data_specs is None:
            data_specs = DataSpecs(
                ImageSpecs(224, mean=ImageSpecs.IMAGENET_MEAN, stddev=ImageSpecs.IMAGENET_STDDEV),
                JointsSpecs(MpiiSkeletonDesc, n_dims=2),
            )

        super().__init__(data_specs)

        self.subset = subset
        self.use_aug = use_aug
        self.mpii_data = MpiiData(data_dir)
        self.example_ids = self.mpii_data.subset_indices(self.subset)[:max_length]

    def to_canonical_skeleton(self, skel, force=False):
        if not force and self.skeleton_desc.canonical:
            return skel

        canonical_joints = [
            MpiiSkeletonDesc.joint_names.index(s if s != 'head' else 'head_top')
            for s in CanonicalSkeletonDesc.joint_names
        ]

        size = list(skel.size())
        size[-2] = len(canonical_joints)
        canonical_joints_tensor = torch.LongTensor(canonical_joints).unsqueeze(-1).expand(size)
        canonical_skel = skel.gather(-2, canonical_joints_tensor)

        # There is no 'head' joint in MPII, so we will interpolate between
        # 'head_top' and 'neck'
        canonical_skel[CanonicalSkeletonDesc.joint_names.index('head')] = (
            0.5 * skel[MpiiSkeletonDesc.joint_names.index('head_top')] +
            0.5 * skel[MpiiSkeletonDesc.joint_names.index('neck')]
        )

        # The 'spine' joint in MPII is close to the neck, not in the middle of the back.
        # Therefore we need to move it closer to the pelvis.
        canonical_skel[CanonicalSkeletonDesc.joint_names.index('spine')] = (
            0.53 * skel[MpiiSkeletonDesc.joint_names.index('spine')] +
            0.47 * skel[MpiiSkeletonDesc.joint_names.index('pelvis')]
        )

        return canonical_skel

    def to_canonical_mask(self, mask, force=False):
        if not force and self.skeleton_desc.canonical:
            return mask

        canonical_joints = [
            MpiiSkeletonDesc.joint_names.index(s if s != 'head' else 'head_top')
            for s in CanonicalSkeletonDesc.joint_names
        ]

        size = list(mask.size())
        size[-1] = len(canonical_joints)
        canonical_joints_tensor = torch.LongTensor(canonical_joints).expand(size)
        canonical_mask = mask.gather(-1, canonical_joints_tensor)
        if mask[MpiiSkeletonDesc.joint_names.index('head_top')] == 0 \
                or mask[MpiiSkeletonDesc.joint_names.index('neck')] == 0:
            canonical_mask[CanonicalSkeletonDesc.joint_names.index('head')] = 0
        else:
            canonical_mask[CanonicalSkeletonDesc.joint_names.index('head')] = 1

        return canonical_mask

    def __len__(self):
        return len(self.example_ids)

    def __getitem__(self, index):
        id = self.example_ids[index]

        normalize = self.mpii_data.head_lengths[id]
        orig_target = torch.from_numpy(self.mpii_data.keypoints[id])
        joint_mask = torch.from_numpy(self.mpii_data.keypoint_masks[id])

        aug_hflip = False
        aug_brightness = aug_contrast = aug_saturation = 1.0
        aug_hue = 0.0
        aug_scale = 1
        aug_rot = 0

        if self.use_aug:
            aug_hflip = np.random.uniform() < 0.5
            if np.random.uniform() < 0.3:
                aug_brightness = np.random.uniform(0.8, 1.2)
            if np.random.uniform() < 0.3:
                aug_contrast = np.random.uniform(0.8, 1.2)
            if np.random.uniform() < 0.3:
                aug_saturation = np.random.uniform(0.8, 1.2)
            if np.random.uniform() < 0.3:
                aug_hue = np.random.uniform(-0.1, 0.1)
            aug_scale = 2 ** np.clip(np.random.normal(0, 0.25), -0.5, 0.5)
            if np.random.uniform() < 0.4:
                aug_rot = np.clip(np.random.normal(0, 30), -60, 60)

        # Get the centre and size of the bounding box
        bb = self.mpii_data.get_bounding_box(id)
        bb_cx = (bb[0] + bb[2]) / 2
        bb_cy = (bb[1] + bb[3]) / 2
        bb_size = bb[2] - bb[0]

        orig_image = self.mpii_data.load_image(id)
        img_short_side = min(orig_image.height, orig_image.width)

        # We don't actually have the camera calibration, so we'll just guess
        # a somewhat sensible focal length. This rough approximation will
        # lead to particularly prominent errors when the subject is not in
        # the original image centre.
        focal_length = orig_image.width * 1.2
        orig_camera = CameraIntrinsics.from_ccd_params(focal_length, focal_length,
                                                       orig_image.width / 2, orig_image.height / 2)
        extrinsics = torch.eye(4, dtype=torch.float64)

        transform_opts = {
            'in_camera': orig_camera,
            'in_width': orig_image.width,
            'in_height': orig_image.height,
            'centre_x': bb_cx,
            'centre_y': bb_cy,
            'rotation': aug_rot,
            'scale': aug_scale * bb_size / img_short_side,
            'hflip_indices': self.skeleton_desc.hflip_indices,
            'hflip': aug_hflip,
            'out_width': self.data_specs.input_specs.width,
            'out_height': self.data_specs.input_specs.height,
            'brightness': aug_brightness,
            'contrast': aug_contrast,
            'saturation': aug_saturation,
            'hue': aug_hue,
        }

        # If a canonical skeleton is expected, convert the keypoints and masks appropriately
        if self.skeleton_desc.canonical:
            orig_target = self.to_canonical_skeleton(orig_target, force=True)
            joint_mask = self.to_canonical_mask(joint_mask, force=True)

        # Tweak the original coordinates to look like they are in 3D camera space
        orig_target = torch.cat(
            [orig_target, torch.ones_like(orig_target.narrow(-1, 0, 2))], -1)
        orig_target[:, 0] -= orig_image.width / 2
        orig_target[:, 1] -= orig_image.height / 2
        orig_target[:, 2] = focal_length

        ctx = self.create_transformer_context(transform_opts)
        camera_int, img, part_coords = ctx.transform(orig_camera, orig_image, orig_target)

        # Normalize coordinates to [-1, 1] range
        z_ref = part_coords[self.skeleton_desc.root_joint_id, 2]
        part_coords = self.skeleton_normaliser.normalise_skeleton(
            part_coords, z_ref, camera_int, img.height, img.width)

        if aug_hflip:
            hflip_indices = torch.LongTensor(self.skeleton_desc.hflip_indices)
            joint_mask.scatter_(0, hflip_indices, joint_mask.clone())

        # Mask out joints that have been transformed to a location outside of the
        # image bounds.
        #
        # NOTE: It is still possible for joints to be transformed outside of the image bounds
        # when augmentations are turned off. This is because the center/scale information
        # provided in the MPII human pose dataset will occasionally not include a joint.
        # For example, see the head top joint for ID 21 in the validation set.
        if self.subset == 'train' or self.subset == 'trainval':
            within_bounds, _ = part_coords.narrow(-1, 0, 2).abs().lt(1).min(-1, keepdim=False)
            joint_mask.mul_(within_bounds)

        sample = {
            'index': index,  # Index in the dataset
            'valid_depth': 0,

            'normalize': normalize,
            'joint_mask': joint_mask,

            'input': self.input_to_tensor(img),
            'camera_intrinsic': camera_int,
            'camera_extrinsic': extrinsics,

            # Transformer data
            'transform_opts': transform_opts,

            'original_skel': orig_target,
            'target': part_coords,
        }

        return sample
