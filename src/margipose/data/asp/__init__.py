"""
Data loader for the ASPset dataset.
"""

from os import path

import aspset.dataset.constants as aspset_constants
import numpy as np
import torch
import torch.nn.functional
from aspset.dataset import AspsetData
from pose3d_utils.camera import CameraIntrinsics
from pose3d_utils.coords import ensure_homogeneous, homogeneous_to_cartesian
from posekit.skeleton import skeleton_converter, skeleton_registry
from torchvision.transforms.functional import to_pil_image

from margipose.data import PoseDataset, collate
from margipose.data.skeleton import SkeletonDesc
from margipose.data_specs import DataSpecs, ImageSpecs, JointsSpecs
from margipose.eval import prepare_for_3d_evaluation, gather_3d_metrics

AspsetSkeletonDesc = SkeletonDesc(
    joint_names=skeleton_registry['aspset_17j'].joint_names,
    joint_tree=skeleton_registry['aspset_17j'].joint_tree,
    hflip_indices=skeleton_registry['aspset_17j'].hflip_indices,
)


def aspset_to_canonical_skeleton(skel):
    """Convert from the ASPset skeleton description to canonical (MPI-INF-3DHP 17-joint)."""
    return skeleton_converter.convert(skel, 'aspset_17j', 'mpi3d_17j')


class AspsetDataset(PoseDataset):
    '''Create a Dataset object for loading ASPset data.

    Args:
        data_dir (str): path to the data directory
        data_specs (DataSpecs):
        subset (str): subset of the data to load ("train", "val", "trainval", or "test")
        use_aug (bool): set to `True` to enable random data augmentation
    '''

    def __init__(self, data_dir, data_specs=None, subset='train', use_aug=False):
        if data_specs is None:
            data_specs = DataSpecs(
                ImageSpecs(224, mean=ImageSpecs.IMAGENET_MEAN, stddev=ImageSpecs.IMAGENET_STDDEV),
                JointsSpecs(AspsetSkeletonDesc, n_dims=2),
            )

        super().__init__(data_specs)

        if not path.isdir(data_dir):
            raise NotADirectoryError(data_dir)

        self.subset = subset
        self.data_dir = data_dir
        self.use_aug = use_aug
        self.resources = AspsetData(data_dir, subset=subset)

        self.without_image = False
        self.multicrop = False

    def to_canonical_skeleton(self, skel):
        if self.skeleton_desc.canonical:
            return skel
        return aspset_to_canonical_skeleton(skel)

    def get_orig_skeleton(self, index):
        joints_3d, skeleton_name = self.resources.get_original_joints_3d(index)
        original_skel = ensure_homogeneous(torch.as_tensor(joints_3d, dtype=torch.float64), d=3)
        if self.skeleton_desc.canonical:
            if skeleton_name == 'aspset_17j' and original_skel.size(-2) == AspsetSkeletonDesc.n_joints:
                original_skel = aspset_to_canonical_skeleton(original_skel)
            else:
                raise Exception('unexpected number of joints: ' + original_skel.size(-2))
        return original_skel

    def _load_image(self, index):
        if self.without_image:
            return None
        image_tensor = self.resources.load_full_image(index)
        return to_pil_image(image_tensor)

    def _evaluate_3d(self, index, original_skel, norm_pred, camera_intrinsics, transform_opts):
        assert self.skeleton_desc.canonical, 'can only evaluate canonical skeletons'
        expected, actual = prepare_for_3d_evaluation(original_skel, norm_pred, self,
                                                     camera_intrinsics, transform_opts,
                                                     known_depth=True)
        return gather_3d_metrics(expected, actual)

    def __len__(self):
        return len(self.resources)

    def _build_sample(self, index, orig_camera, orig_image, orig_skel, transform_opts, extrinsics):
        out_width = self.data_specs.input_specs.width
        out_height = self.data_specs.input_specs.height

        ctx = self.create_transformer_context(transform_opts)
        camera_int, img, joints3d = ctx.transform(orig_camera, orig_image, orig_skel)

        z_ref = joints3d[self.skeleton_desc.root_joint_id, 2]
        target = self.skeleton_normaliser.normalise_skeleton(
            joints3d, z_ref, camera_int, out_height, out_width)

        sample = {
            'index': index,  # Index in the dataset
            'valid_depth': 1,

            # "Original" data without transforms applied
            'original_skel': orig_skel,

            # Transformed data
            'camera_intrinsic': camera_int,
            'camera_extrinsic': extrinsics,
            'target': target,  # Normalised target skeleton

            # Transformer data
            'transform_opts': transform_opts,

            'joint_mask': torch.ones(target.size(-2), dtype=torch.uint8),
        }

        if img:
            sample['input'] = self.input_to_tensor(img)

        return sample

    def __getitem__(self, index):
        orig_image = self._load_image(index)
        if orig_image:
            img_w, img_h = orig_image.size
        else:
            img_w = aspset_constants.FULL_IMAGE_WIDTH
            img_h = aspset_constants.FULL_IMAGE_HEIGHT
        img_short_side = min(img_h, img_w)

        extrinsics = torch.eye(4).double()
        extrinsics[:3, :4] = torch.as_tensor(self.resources.get_camera_extrinsics(index), dtype=torch.float64)
        intrinsics = torch.zeros(3, 4)
        intrinsics[:3, :3] = torch.as_tensor(self.resources.get_camera_intrinsics(index), dtype=torch.float64)
        orig_camera = CameraIntrinsics(intrinsics)

        orig_skel = self.get_orig_skeleton(index)

        # Bounding box details
        joints2d = homogeneous_to_cartesian(
            orig_camera.project(ensure_homogeneous(orig_skel, d=3)))
        min_x = joints2d[:, 0].min().item()
        max_x = joints2d[:, 0].max().item()
        min_y = joints2d[:, 1].min().item()
        max_y = joints2d[:, 1].max().item()
        bb_cx = (min_x + max_x) / 2
        bb_cy = (min_y + max_y) / 2
        bb_size = 1.5 * max(max_x - min_x, max_y - min_y)

        out_width = self.data_specs.input_specs.width
        out_height = self.data_specs.input_specs.height

        if self.multicrop:
            samples = []
            for aug_hflip in [False, True]:
                for offset in [(0, 0), (-1, 0), (0, -1), (1, 0), (0, 1)]:
                    aug_x = offset[0] * 8
                    aug_y = offset[1] * 8

                    transform_opts = {
                        'in_camera': orig_camera,
                        'in_width': img_w,
                        'in_height': img_h,
                        'centre_x': bb_cx + aug_x,
                        'centre_y': bb_cy + aug_y,
                        'rotation': 0,
                        'scale': bb_size / img_short_side,
                        'hflip_indices': self.skeleton_desc.hflip_indices,
                        'hflip': aug_hflip,
                        'out_width': out_width,
                        'out_height': out_height,
                        'brightness': 1,
                        'contrast': 1,
                        'saturation': 1,
                        'hue': 0,
                    }

                    samples.append(self._build_sample(index, orig_camera, orig_image, orig_skel,
                                                      transform_opts, extrinsics))

            return collate(samples)
        else:
            aug_hflip = False
            aug_brightness = aug_contrast = aug_saturation = 1.0
            aug_hue = 0.0
            aug_x = aug_y = 0.0
            aug_scale = 1.0
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
                aug_x = np.random.uniform(-16, 16)
                aug_y = np.random.uniform(-16, 16)
                aug_scale = np.random.uniform(0.9, 1.1)
                if np.random.uniform() < 0.4:
                    aug_rot = np.clip(np.random.normal(0, 30), -30, 30)

            transform_opts = {
                'in_camera': orig_camera,
                'in_width': img_w,
                'in_height': img_h,
                'centre_x': bb_cx + aug_x,
                'centre_y': bb_cy + aug_y,
                'rotation': aug_rot,
                'scale': bb_size * aug_scale / img_short_side,
                'hflip_indices': self.skeleton_desc.hflip_indices,
                'hflip': aug_hflip,
                'out_width': out_width,
                'out_height': out_height,
                'brightness': aug_brightness,
                'contrast': aug_contrast,
                'saturation': aug_saturation,
                'hue': aug_hue,
            }

            return self._build_sample(index, orig_camera, orig_image, orig_skel, transform_opts,
                                      extrinsics)
