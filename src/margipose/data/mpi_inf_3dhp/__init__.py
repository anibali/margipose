import re
from glob import iglob
from os import path

import h5py
import numpy as np
import torch
from PIL import Image, ImageOps
from pose3d_utils.coords import homogeneous_to_cartesian, ensure_homogeneous
from torchvision.transforms import RandomCrop, RandomHorizontalFlip

from margipose.data import PoseDataset, collate
from margipose.data.mpi_inf_3dhp.common import Annotations, parse_camera_calibration, Constants, \
    MpiInf3dhpSkeletonDesc
from margipose.data.skeleton import CanonicalSkeletonDesc, VNect_Common_Skeleton
from margipose.data_specs import DataSpecs, ImageSpecs, JointsSpecs
from margipose.eval import prepare_for_3d_evaluation, gather_3d_metrics


class FrameRef:
    def __init__(self, subject_id, sequence_id, camera_id, frame_index, activity_id=None):
        self.subject_id = subject_id
        self.sequence_id = sequence_id
        self.camera_id = camera_id
        self.frame_index = frame_index
        self.activity_id = activity_id

    @property
    def image_file(self):
        return 'S{}/Seq{}/imageSequence/video_{}/img_{:06d}.jpg'.format(
            self.subject_id, self.sequence_id, self.camera_id, self.frame_index + 1
        )

    @property
    def bg_mask_file(self):
        return 'S{}/Seq{}/foreground_mask/video_{}/img_{:06d}.png'.format(
            self.subject_id, self.sequence_id, self.camera_id, self.frame_index + 1
        )

    @property
    def ub_mask_file(self):
        return 'S{}/Seq{}/up_body_mask/video_{}/img_{:06d}.png'.format(
            self.subject_id, self.sequence_id, self.camera_id, self.frame_index + 1
        )

    @property
    def lb_mask_file(self):
        return 'S{}/Seq{}/low_body_mask/video_{}/img_{:06d}.png'.format(
            self.subject_id, self.sequence_id, self.camera_id, self.frame_index + 1
        )

    @property
    def annot_file(self):
        return 'S{}/Seq{}/annot.mat'.format(self.subject_id, self.sequence_id)

    @property
    def camera_file(self):
        return 'S{}/Seq{}/camera.calibration'.format(self.subject_id, self.sequence_id)

    @property
    def metadata_file(self):
        return 'S{}/Seq{}/metadata.h5'.format(self.subject_id, self.sequence_id)

    @property
    def bg_augmentable(self):
        seq_path = 'S{}/Seq{}'.format(self.subject_id, self.sequence_id)
        return Constants['seq_info'][seq_path]['bg_augmentable'] == 1

    @property
    def ub_augmentable(self):
        seq_path = 'S{}/Seq{}'.format(self.subject_id, self.sequence_id)
        return Constants['seq_info'][seq_path]['ub_augmentable'] == 1

    @property
    def lb_augmentable(self):
        seq_path = 'S{}/Seq{}'.format(self.subject_id, self.sequence_id)
        return Constants['seq_info'][seq_path]['lb_augmentable'] == 1

    def to_dict(self):
        return {
            'subject_id': self.subject_id,
            'sequence_id': self.sequence_id,
            'camera_id': self.camera_id,
            'frame_index': self.frame_index,
            'activity_id': self.activity_id,
        }


def random_texture():
    files = list(iglob('resources/textures/*.png'))
    file = files[np.random.randint(0, len(files))]
    texture = Image.open(file).convert('L')
    texture = ImageOps.colorize(
        texture,
        'black',
        (np.random.randint(50, 256), np.random.randint(50, 256), np.random.randint(50, 256))
    )
    return texture


def augment_clothing(img, mask, texture):
    a = np.array(img)
    grey = a.mean(axis=-1)
    blackness = (255 - grey).clip(min=0) / 255

    texture = np.array(texture, dtype=np.float)
    texture -= blackness[..., np.newaxis] * texture
    texture = Image.fromarray(texture.round().astype(np.uint8))

    return Image.composite(texture, img, mask)


def random_background():
    files = list(iglob('resources/backgrounds/*.jpg'))
    file = files[np.random.randint(0, len(files))]
    bg = Image.open(file)
    bg = RandomHorizontalFlip()(RandomCrop(768)(bg))
    return bg


def augment_background(img, mask, bg):
    return Image.composite(img, bg, mask)


class MpiInf3dDataset(PoseDataset):
    preserve_root_joint_at_univ_scale = False

    def __init__(self, data_dir, data_specs=None, use_aug=False, disable_mask_aug=False):
        if data_specs is None:
            data_specs = DataSpecs(
                ImageSpecs(224, mean=ImageSpecs.IMAGENET_MEAN, stddev=ImageSpecs.IMAGENET_STDDEV),
                JointsSpecs(MpiInf3dhpSkeletonDesc, n_dims=3),
            )

        super().__init__(data_specs)

        if not path.isdir(data_dir):
            raise NotADirectoryError(data_dir)

        metadata_files = sorted(iglob(path.join(data_dir, 'S*', 'Seq*', 'metadata.h5')))
        frame_refs = []
        univ_scale_factors = {}

        for metadata_file in metadata_files:
            match = re.match(r'.*S(\d+)/Seq(\d+)/metadata.h5', metadata_file)
            subject_id = int(match.group(1))
            sequence_id = int(match.group(2))

            activity_ids = None
            mat_annot_file = path.join(path.dirname(metadata_file), 'annot_data.mat')
            if path.isfile(mat_annot_file):
                with h5py.File(mat_annot_file, 'r') as f:
                    activity_ids = f['activity_annotation'][:].flatten().astype(int)

            with h5py.File(metadata_file, 'r') as f:
                keys = f['interesting_frames'].keys()
                for key in keys:
                    camera_id = int(re.match(r'camera(\d)', key).group(1))
                    for frame_index in f['interesting_frames'][key]:
                        activity_id = None
                        if activity_ids is not None:
                            activity_id = activity_ids[frame_index]
                        frame_refs.append(FrameRef(subject_id, sequence_id, camera_id, frame_index, activity_id))
                univ_scale_factors[(subject_id, sequence_id)] = f['scale'][0]

        self.data_dir = data_dir
        self.use_aug = use_aug
        self.disable_mask_aug = disable_mask_aug
        self.frame_refs = frame_refs
        self.univ_scale_factors = univ_scale_factors
        self.without_image = False
        self.multicrop = False

    @staticmethod
    def _mpi_inf_3dhp_to_canonical_skeleton(skel):
        assert skel.size(-2) == MpiInf3dhpSkeletonDesc.n_joints

        canonical_joints = [
            MpiInf3dhpSkeletonDesc.joint_names.index(s)
            for s in CanonicalSkeletonDesc.joint_names
        ]
        size = list(skel.size())
        size[-2] = len(canonical_joints)
        canonical_joints_tensor = torch.LongTensor(canonical_joints).unsqueeze(-1).expand(size)
        return skel.gather(-2, canonical_joints_tensor)

    def to_canonical_skeleton(self, skel):
        if self.skeleton_desc.canonical:
            return skel

        return self._mpi_inf_3dhp_to_canonical_skeleton(skel)

    def _get_skeleton_3d(self, index):
        frame_ref = self.frame_refs[index]
        metadata_file = path.join(self.data_dir, frame_ref.metadata_file)
        with h5py.File(metadata_file, 'r') as f:
            # Load the pose joint locations
            original_skel = torch.from_numpy(
                f['joints3d'][frame_ref.camera_id, frame_ref.frame_index]
            )

        if original_skel.shape[-2] == MpiInf3dhpSkeletonDesc.n_joints:
            # The training/validation skeletons have 28 joints.
            skel_desc = MpiInf3dhpSkeletonDesc
        elif original_skel.shape[-2] == CanonicalSkeletonDesc.n_joints:
            # The test set skeletons have the 17 canonical joints only.
            skel_desc = CanonicalSkeletonDesc
        else:
            raise Exception('unexpected number of joints: ' + original_skel.shape[-2])

        if self.skeleton_desc.canonical:
            if skel_desc == MpiInf3dhpSkeletonDesc:
                original_skel = self._mpi_inf_3dhp_to_canonical_skeleton(original_skel)
            elif skel_desc == CanonicalSkeletonDesc:
                # No conversion necessary.
                pass
            else:
                raise Exception()
            skel_desc = CanonicalSkeletonDesc

        return original_skel, skel_desc

    def _to_univ_scale(self, skel_3d, skel_desc, univ_scale_factor):
        univ_skel_3d = skel_3d.clone()

        # Scale the skeleton to match the universal skeleton size
        if self.preserve_root_joint_at_univ_scale:
            # Scale the skeleton about the root joint position. This should give the same
            # joint position coordinates as the "univ_annot3" annotations.
            root = skel_3d[..., skel_desc.root_joint_id:skel_desc.root_joint_id+1, :]
            univ_skel_3d -= root
            univ_skel_3d /= univ_scale_factor
            univ_skel_3d += root
        else:
            # Scale the skeleton about the camera position. Useful for breaking depth/scale
            # ambiguity.
            univ_skel_3d /= univ_scale_factor

        return univ_skel_3d

    def _evaluate_3d(self, index, original_skel, norm_pred, camera_intrinsics, transform_opts):
        assert self.skeleton_desc.canonical, 'can only evaluate canonical skeletons'
        expected, actual = prepare_for_3d_evaluation(original_skel, norm_pred, self,
                                                     camera_intrinsics, transform_opts,
                                                     known_depth=False)
        included_joints = [
            CanonicalSkeletonDesc.joint_names.index(joint_name)
            for joint_name in VNect_Common_Skeleton
        ]
        return gather_3d_metrics(expected, actual, included_joints)

    def __len__(self):
        return len(self.frame_refs)

    def _build_sample(self, index, orig_camera, orig_image, orig_skel, transform_opts, extrinsics):
        frame_ref = self.frame_refs[index]
        out_width = self.data_specs.input_specs.width
        out_height = self.data_specs.input_specs.height

        ctx = self.create_transformer_context(transform_opts)
        camera_int, img, joints3d = ctx.transform(orig_camera, orig_image, orig_skel)

        z_ref = joints3d[self.skeleton_desc.root_joint_id, 2]
        target = self.skeleton_normaliser.normalise_skeleton(
            joints3d, z_ref, camera_int, out_height, out_width)

        sample = {
            # Description of which video frame the example comes from
            'frame_ref': frame_ref.to_dict(),
            'index': index,  # Index in the dataset
            'valid_depth': 1,

            # "Original" data without transforms applied
            'original_skel': ensure_homogeneous(orig_skel, d=3),  # Universal scale

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
        frame_ref = self.frame_refs[index]

        skel_3d, skel_desc = self._get_skeleton_3d(index)
        univ_scale_factor = self.univ_scale_factors[(frame_ref.subject_id, frame_ref.sequence_id)]
        orig_skel = self._to_univ_scale(skel_3d, skel_desc, univ_scale_factor)

        if self.without_image:
            orig_image = None
            img_w = img_h = 768
        else:
            orig_image = Image.open(path.join(self.data_dir, frame_ref.image_file))
            img_w, img_h = orig_image.size

        with open(path.join(self.data_dir, frame_ref.camera_file), 'r') as f:
            cam_cal = parse_camera_calibration(f)[frame_ref.camera_id]

        # Correct the camera to account for the fact that video frames were
        # stored at a lower resolution.
        orig_camera = cam_cal['intrinsics'].clone()
        old_w = cam_cal['image_width']
        old_h = cam_cal['image_height']
        orig_camera.scale_image(img_w / old_w, img_h / old_h)

        extrinsics = cam_cal['extrinsics']

        # Bounding box details
        skel_2d = orig_camera.project_cartesian(skel_3d)
        min_x = skel_2d[:, 0].min().item()
        max_x = skel_2d[:, 0].max().item()
        min_y = skel_2d[:, 1].min().item()
        max_y = skel_2d[:, 1].max().item()
        bb_cx = (min_x + max_x) / 2
        bb_cy = (min_y + max_y) / 2
        bb_size = 1.5 * max(max_x - min_x, max_y - min_y)

        img_short_side = min(img_h, img_w)
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
            aug_bg = aug_ub = aug_lb = False
            aug_hflip = False
            aug_brightness = aug_contrast = aug_saturation = 1.0
            aug_hue = 0.0
            aug_x = aug_y = 0.0
            aug_scale = 1.0
            aug_rot = 0

            if self.use_aug:
                if not self.disable_mask_aug:
                    aug_bg = frame_ref.bg_augmentable and np.random.uniform() < 0.6
                    aug_ub = frame_ref.ub_augmentable and np.random.uniform() < 0.2
                    aug_lb = frame_ref.lb_augmentable and np.random.uniform() < 0.5
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

            if orig_image:
                if aug_bg:
                    orig_image = augment_background(
                        orig_image,
                        Image.open(path.join(self.data_dir, frame_ref.bg_mask_file)),
                        random_background()
                    )
                if aug_ub:
                    orig_image = augment_clothing(
                        orig_image,
                        Image.open(path.join(self.data_dir, frame_ref.ub_mask_file)),
                        random_texture()
                    )
                if aug_lb:
                    orig_image = augment_clothing(
                        orig_image,
                        Image.open(path.join(self.data_dir, frame_ref.lb_mask_file)),
                        random_texture()
                    )

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
