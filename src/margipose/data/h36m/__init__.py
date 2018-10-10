"""
Data loader for the Human 3.6M dataset.

Dataset home page: http://vision.imar.ro/human3.6m/
"""

from glob import iglob
from os import path

import h5py
import numpy as np
import torch
import torch.nn.functional
from PIL import Image
from pose3d_utils.camera import CameraIntrinsics
from pose3d_utils.coords import ensure_homogeneous, homogeneous_to_cartesian

from margipose.data import PoseDataset, collate
from margipose.data.skeleton import CanonicalSkeletonDesc, SkeletonDesc
from margipose.data_specs import DataSpecs, ImageSpecs, JointsSpecs
from margipose.eval import prepare_for_3d_evaluation, gather_3d_metrics

H36MSkeletonDesc = SkeletonDesc(
    joint_names=[
        # 0-3
        'pelvis', 'right_hip', 'right_knee', 'right_ankle',
        # 4-7
        'right_toes', 'right_site1', 'left_hip', 'left_knee',
        # 8-11
        'left_ankle', 'left_toes', 'left_site1', 'spine1',
        # 12-15
        'spine', 'neck', 'head', 'head_top',
        # 16-19
        'left_clavicle', 'left_shoulder', 'left_elbow', 'left_wrist',
        # 20-23
        'left_thumb', 'left_site2', 'left_wrist2', 'left_site3',
        # 24-27
        'right_clavicle', 'right_shoulder', 'right_elbow', 'right_wrist',
        # 28-31
        'right_thumb', 'right_site2', 'right_wrist2', 'right_site3'
    ],
    joint_tree=[
        0, 0, 1, 2,
        3, 4, 0, 6,
        7, 8, 9, 0,
        11, 12, 13, 14,
        12, 16, 17, 18,
        19, 20, 19, 22,
        12, 24, 25, 26,
        27, 28, 27, 30,
    ],
    hflip_indices=[
        0, 6, 7, 8,
        9, 10, 1, 2,
        3, 4, 5, 11,
        12, 13, 14, 15,
        24, 25, 26, 27,
        28, 29, 30, 31,
        16, 17, 18, 19,
        20, 21, 22, 23,
    ]
)

H36M_Actions = {
    1:  'Miscellaneous',
    2:  'Directions',
    3:  'Discussion',
    4:  'Eating',
    5:  'Greeting',
    6:  'Phoning',
    7:  'Posing',
    8:  'Purchases',
    9:  'Sitting',
    10: 'SittingDown',
    11: 'Smoking',
    12: 'TakingPhoto',
    13: 'Waiting',
    14: 'Walking',
    15: 'WalkingDog',
    16: 'WalkingTogether',
}


def h36m_to_canonical_skeleton(skel):
    assert skel.size(-2) == H36MSkeletonDesc.n_joints

    canonical_joints = [
        H36MSkeletonDesc.joint_names.index(s)
        for s in CanonicalSkeletonDesc.joint_names
    ]
    size = list(skel.size())
    size[-2] = len(canonical_joints)
    canonical_joints_tensor = torch.LongTensor(canonical_joints).unsqueeze(-1).expand(size)
    return skel.gather(-2, canonical_joints_tensor)


class H36MDataset(PoseDataset):
    '''Create a Dataset object for loading Human 3.6M pose data (protocol 2).

    Args:
        data_dir (str): path to the data directory
        data_specs (DataSpecs):
        subset (str): subset of the data to load ("train", "val", "trainval", or "test")
        use_aug (bool): set to `True` to enable random data augmentation
        max_length (int):
        universal (bool): set to `True` to use universal skeleton scale
    '''

    def __init__(self, data_dir, data_specs=None, subset='train', use_aug=False, max_length=None,
                 universal=False):
        if data_specs is None:
            data_specs = DataSpecs(
                ImageSpecs(224, mean=ImageSpecs.IMAGENET_MEAN, stddev=ImageSpecs.IMAGENET_STDDEV),
                JointsSpecs(H36MSkeletonDesc, n_dims=2),
            )

        super().__init__(data_specs)

        if not path.isdir(data_dir):
            raise NotADirectoryError(data_dir)

        self.subset = subset
        self.use_aug = use_aug
        self.data_dir = data_dir

        annot_files = sorted(iglob(path.join(data_dir, 'S*', '*', 'annot.h5')))

        keys = ['pose/2d', 'pose/3d', 'pose/3d-univ', 'camera', 'frame',
                'subject', 'action', 'subaction']
        datasets = {}
        self.camera_intrinsics = []

        intrinsics_ds = 'intrinsics-univ' if universal else 'intrinsics'

        for annot_file in annot_files:
            with h5py.File(annot_file) as annot:
                for k in keys:
                    if k in datasets:
                        datasets[k].append(annot[k].value)
                    else:
                        datasets[k] = [annot[k].value]
                cams = {}
                for camera_id in annot[intrinsics_ds].keys():
                    alpha_x, x_0, alpha_y, y_0 = list(annot[intrinsics_ds][camera_id])
                    cams[int(camera_id)] = CameraIntrinsics.from_ccd_params(alpha_x, alpha_y, x_0, y_0)
                for camera_id in annot['camera']:
                    self.camera_intrinsics.append(cams[camera_id])
        datasets = {k: np.concatenate(v) for k, v in datasets.items()}

        self.frame_ids = datasets['frame']
        self.subject_ids = datasets['subject']
        self.action_ids = datasets['action']
        self.subaction_ids = datasets['subaction']
        self.camera_ids = datasets['camera']
        self.joint_3d = datasets['pose/3d-univ'] if universal else datasets['pose/3d']
        self.joint_2d = datasets['pose/2d']

        # Protocol #2
        train_subjects = {1, 5, 6, 7, 8}
        test_subjects = {9, 11}

        train_ids = []
        test_ids = []

        for index, subject_id in enumerate(self.subject_ids):
            if subject_id in train_subjects:
                train_ids.append(index)
            if subject_id in test_subjects:
                test_ids.append(index)

        if subset == 'trainval':
            self.example_ids = np.array(train_ids, np.uint32)
        elif subset == 'test':
            self.example_ids = np.array(test_ids, np.uint32)
        else:
            raise Exception('Only trainval and test subsets are supported')

        if max_length is not None:
            self.example_ids = self.example_ids[:max_length]

        self.without_image = False
        self.multicrop = False

    def to_canonical_skeleton(self, skel):
        if self.skeleton_desc.canonical:
            return skel
        return h36m_to_canonical_skeleton(skel)

    def get_orig_skeleton(self, index):
        id = self.example_ids[index]
        original_skel = ensure_homogeneous(torch.from_numpy(self.joint_3d[id]), d=3)
        if self.skeleton_desc.canonical:
            if original_skel.size(-2) == H36MSkeletonDesc.n_joints:
                original_skel = h36m_to_canonical_skeleton(original_skel)
            else:
                raise Exception('unexpected number of joints: ' + original_skel.size(-2))
        return original_skel

    def _load_image(self, id):
        if self.without_image:
            return None
        image_file = path.join(
            self.data_dir,
            'S{:d}'.format(self.subject_ids[id]),
            '{}-{:d}'.format(H36M_Actions[self.action_ids[id]], self.subaction_ids[id]),
            'imageSequence',
            str(self.camera_ids[id]),
            'img_{:06d}.jpg'.format(self.frame_ids[id])
        )
        return Image.open(image_file)

    def _evaluate_3d(self, index, original_skel, norm_pred, camera_intrinsics, transform_opts):
        assert self.skeleton_desc.canonical, 'can only evaluate canonical skeletons'
        expected, actual = prepare_for_3d_evaluation(original_skel, norm_pred, self,
                                                     camera_intrinsics, transform_opts,
                                                     known_depth=True)
        return gather_3d_metrics(expected, actual)

    def __len__(self):
        return len(self.example_ids)

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

            'joint_mask': torch.ByteTensor(target.size(-2)).fill_(1),
        }

        if img:
            sample['input'] = self.input_to_tensor(img)

        return sample

    def __getitem__(self, index):
        id = self.example_ids[index]

        orig_image = self._load_image(id)
        if orig_image:
            img_w, img_h = orig_image.size
        else:
            img_w = img_h = 1000
        img_short_side = min(img_h, img_w)

        extrinsics = torch.eye(4).double()
        orig_camera = self.camera_intrinsics[id]

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
