"""
Data loader for the MPII 2D human pose dataset.

Dataset home page: http://human-pose.mpi-inf.mpg.de/
"""

from os import path

import h5py
import numpy as np
import torch
import torch.nn.functional
from PIL import Image
from pose3d_utils.camera import CameraIntrinsics

from margipose.data import PoseDataset
from margipose.data.skeleton import SkeletonDesc, CanonicalSkeletonDesc
from margipose.data_specs import DataSpecs, ImageSpecs, JointsSpecs

MpiiSkeletonDesc = SkeletonDesc(
    joint_names=[
        # 0-3
        'right_ankle', 'right_knee', 'right_hip', 'left_hip',
        # 4-7
        'left_knee', 'left_ankle', 'pelvis', 'spine',
        # 8-11
        'neck', 'head_top', 'right_wrist', 'right_elbow',
        # 12-15
        'right_shoulder', 'left_shoulder', 'left_elbow', 'left_wrist'
    ],
    joint_tree=[
        1, 2, 6, 6,
        3, 4, 6, 6,
        7, 8, 11, 12,
        8, 8, 13, 14
    ],
    hflip_indices=[
        5, 4, 3, 2,
        1, 0, 6, 7,
        8, 9, 15, 14,
        13, 12, 11, 10
    ]
)


class MpiiDataset(PoseDataset):
    '''Create a Dataset object for loading MPII Human Pose data.

    Here's where you can get all of the files:

    `images/`
    - url: http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz
    - md5sum: b6bc9c6869d3f035a5570b2e68ec84c4

    `mpii_annot_all.h5`
    - url: https://github.com/anewell/pose-hg-train/raw/4637618a1b162d80436bfd0b557833b5824cbb21/data/mpii/annot.h5
    - md5sum: c0d0ba453709e37d632b4d4059e2799c

    `mpii_annot_valid.h5`
    - url: https://github.com/anewell/pose-hg-train/raw/4637618a1b162d80436bfd0b557833b5824cbb21/data/mpii/annot/valid.h5
    - md5sum: d88b6828485168c1fb4c79a21995fdef

    Args:
        data_dir (str): path to the directory containing `images/` and `mpii_annot_*.h5`
        data_specs (DataSpecs):
        subset (str): subset of the data to load ("train", "val", "trainval", or "test")
        use_aug (bool): set to `True` to enable random data augmentation
        max_length (int):
    '''

    def __init__(self, data_dir, data_specs=None, subset='train', use_aug=False, max_length=None):
        if data_specs is None:
            data_specs = DataSpecs(
                ImageSpecs(224, mean=ImageSpecs.IMAGENET_MEAN, stddev=ImageSpecs.IMAGENET_STDDEV),
                JointsSpecs(MpiiSkeletonDesc, n_dims=2, coord_space='square'),
            )

        super().__init__(data_specs)

        self.subset = subset
        self.use_aug = use_aug

        all_annot_file = path.join(data_dir, 'mpii_annot_all.h5')
        val_annot_file = path.join(data_dir, 'mpii_annot_valid.h5')

        with h5py.File(all_annot_file, 'r') as f:
            image_indices = f['/index'].value.astype(np.uint64)
            is_train = f['/istrain'].value
            imgnames = f['/imgname'].value.astype(np.uint8)
            people = f['/person'].value.astype(np.uint64)

            image_paths = [
                path.join(data_dir, 'images', imgname.tostring().decode('UTF-8').split('\0')[0])
                for imgname in imgnames
            ]
            annotations = {
                'centers': f['/center'].value,
                'scales': f['/scale'].value,
                'parts': f['/part'].value,
                'visible': f['/visible'].value,
                'normalize': f['/normalize'].value,
                'image_paths': image_paths,
            }

        with h5py.File(val_annot_file) as f:
            val_image_indices = f['/index'].value.astype(np.uint64)
            val_people = f['/person'].value.astype(np.uint64)

        train_ids = []
        val_ids = []
        test_ids = []

        all_identifiers = np.stack([image_indices, people], axis=-1)
        val_identifiers = np.stack([val_image_indices, val_people], axis=-1)

        for i in range(len(image_indices)):
            val_pos = len(val_ids)
            if val_pos < len(val_image_indices) and np.array_equal(all_identifiers[i], val_identifiers[val_pos]):
                val_ids.append(i)
            elif is_train[i] == 0:
                test_ids.append(i)
            else:
                train_ids.append(i)

        self.annots = annotations
        if subset == 'train':
            self.ids = train_ids
        elif subset == 'val':
            self.ids = val_ids
        elif subset == 'trainval':
            self.ids = train_ids + val_ids
        elif subset == 'test':
            self.ids = test_ids
        else:
            raise Exception('unrecognised subset: {}'.format(subset))

        if max_length is not None and max_length < len(self.ids):
            self.ids = self.ids[:max_length]

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
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]
        image_path = self.annots['image_paths'][id]
        scale = self.annots['scales'][id]
        center = self.annots['centers'][id].copy()
        normalize = self.annots['normalize'][id]

        if self.subset != 'test':
            orig_target = torch.from_numpy(self.annots['parts'][id])
            valid_coords = orig_target.gt(1)
            joint_mask = valid_coords[:, 0] * valid_coords[:, 1]
        else:
            orig_target = None
            joint_mask = torch.ByteTensor(16).fill_(1)

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

        # Small adjustments to reduce the likelihood of cropping out joints
        center[1] += 15 * scale
        scale *= 1.25

        # We will consider this to be the bounding box size for the person
        sz = 200 * scale

        orig_image = Image.open(image_path)
        img_short_side = min(orig_image.height, orig_image.width)

        # We don't actually have the camera calibration, so we'll just guess
        # a somewhat sensible focal length. This rough approximation will
        # lead to particularly prominent errors when the subject is not in
        # the original image centre.
        focal_length = orig_image.width * 1.2
        orig_camera = CameraIntrinsics.from_ccd_params(focal_length, focal_length, 0, 0)
        extrinsics = torch.eye(4).double()

        transform_opts = {
            'in_camera': orig_camera,
            'in_width': orig_image.width,
            'in_height': orig_image.height,
            'centre_x': float(center[0]),
            'centre_y': float(center[1]),
            'rotation': aug_rot,
            'scale': aug_scale * sz / img_short_side,
            'hflip_indices': self.skeleton_desc.hflip_indices,
            'hflip': aug_hflip,
            'out_width': self.data_specs.input_specs.width,
            'out_height': self.data_specs.input_specs.height,
            'brightness': aug_brightness,
            'contrast': aug_contrast,
            'saturation': aug_saturation,
            'hue': aug_hue,
        }

        orig_z_ref = focal_length
        ctx = self.create_transformer_context(transform_opts)
        camera_int, img, _ = ctx.transform(orig_camera, orig_image, None)

        if orig_target is not None:
            if self.skeleton_desc.canonical:
                orig_target = self.to_canonical_skeleton(orig_target, force=True)
                joint_mask = self.to_canonical_mask(joint_mask, force=True)

            orig_target = torch.cat(
                [orig_target, torch.ones_like(orig_target.narrow(-1, 0, 2))], -1)
            orig_target[:, 2] = orig_z_ref
            part_coords = ctx.point_transformer.transform(orig_target)

            # Normalize to [-1, 1] range
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
        }

        if orig_target is not None:
            sample['original_skel'] = orig_target
            sample['target'] = part_coords

        return sample
