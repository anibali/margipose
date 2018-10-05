from abc import ABCMeta, abstractmethod
from collections import Mapping, Sequence

import torch
from pose3d_utils.camera import CameraIntrinsics
from pose3d_utils.coords import ensure_homogeneous
from pose3d_utils.skeleton_normaliser import SkeletonNormaliser
from pose3d_utils.transformers import TransformerContext
import pose3d_utils.transforms as transforms
from torch._six import string_classes, int_classes
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate, DataLoader, SequentialSampler
from torch.utils.data.sampler import WeightedRandomSampler

from margipose.data.skeleton import SkeletonDesc, calculate_knee_neck_height, \
    absolute_to_parent_relative, cartesian_to_spherical, calc_relative_scale, \
    make_eval_scale_bone_lengths, make_eval_scale_skeleton_height
from margipose.data_specs import DataSpecs
from margipose.utils import seed_all


class PoseDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, data_specs: DataSpecs):
        self.data_specs = data_specs
        self.skeleton_normaliser = SkeletonNormaliser()

    def sampler(self, examples_per_epoch=None):
        total_length = len(self)
        if examples_per_epoch is None:
            examples_per_epoch = total_length

        # Sample with replacement only if we have to
        replacement = examples_per_epoch > total_length

        return WeightedRandomSampler(
            torch.ones(total_length).double(),
            examples_per_epoch,
            replacement=replacement
        )

    def input_to_pil_image(self, tensor):
        return self.data_specs.input_specs.unconvert(tensor)

    def input_to_tensor(self, img):
        return self.data_specs.input_specs.convert(img)

    @property
    def skeleton_desc(self) -> SkeletonDesc:
        return self.data_specs.output_specs.skeleton_desc

    def denormalise_with_depth(self, normalised_skel, z_ref, intrinsics):
        """Transforms a normalised skeleton to denormalised form.

        Follow this up with point_transformer.untransform() to get
        a skeleton which is comparable with original_skel.
        """
        return self.skeleton_normaliser.denormalise_skeleton(
            ensure_homogeneous(normalised_skel, d=3),
            z_ref,
            intrinsics,
            self.data_specs.input_specs.height,
            self.data_specs.input_specs.width
        )

    def denormalise(self, normalised_skel, eval_scale, intrinsics):
        """Transforms a normalised skeleton to denormalised form.

        Follow this up with point_transformer.untransform() to get
        a skeleton which is comparable with original_skel.
        """
        normalised_skel = ensure_homogeneous(normalised_skel, d=3)
        z_ref = self.skeleton_normaliser.infer_depth(
            normalised_skel,
            eval_scale,
            intrinsics,
            self.data_specs.input_specs.height,
            self.data_specs.input_specs.width
        )
        return self.denormalise_with_depth(normalised_skel, z_ref, intrinsics)

    def denormalise_with_reference(self, normalised_skel, ref_skel, intrinsics, trans_opts):
        untransform = lambda skel: self.untransform_skeleton(skel, trans_opts)
        eval_scale = make_eval_scale_bone_lengths(self.skeleton_desc, untransform, ref_skel)
        return self.denormalise(normalised_skel, eval_scale, intrinsics)

    def denormalise_with_skeleton_height(self, normalised_skel, intrinsics, trans_opts):
        untransform = lambda skel: self.untransform_skeleton(skel, trans_opts)
        eval_scale = make_eval_scale_skeleton_height(self.skeleton_desc, untransform)
        return self.denormalise(normalised_skel, eval_scale, intrinsics)

    def to_image_space(self, index, normalised, intrinsics):
        z_ref = 100  # Depth value doesn't matter since we are projecting to 2D
        denormalised = self.denormalise_with_depth(normalised, z_ref, intrinsics)
        return intrinsics.project_cartesian(denormalised)

    @staticmethod
    def create_transformer_context(opts) -> TransformerContext:
        ctx = TransformerContext(opts['in_camera'], opts['in_width'], opts['in_height'])
        ctx.add(transforms.PanImage(opts['in_camera'].x_0 - opts['centre_x'], opts['in_camera'].y_0 - opts['centre_y']))
        ctx.add(transforms.RotateImage(opts['rotation']))
        ctx.add(transforms.ZoomImage(1 / opts['scale']))
        ctx.add(transforms.HorizontalFlip(opts['hflip_indices'], opts['hflip']))
        ctx.add(transforms.SquareCrop())
        ctx.add(transforms.ChangeResolution(opts['out_width'], opts['out_height']))
        ctx.add(transforms.AdjustColour(opts['brightness'], opts['contrast'], opts['saturation'], opts['hue']))

        return ctx

    def untransform_skeleton(self, denorm_skel, trans_opts):
        """Transform a denormalised skeleton back into universal camera space."""
        # We can take z_ref from denorm_skel directly because we know that the transformer
        # doesn't change this value.
        ctx = self.create_transformer_context(trans_opts)
        return ctx.point_transformer.untransform(denorm_skel)

    @abstractmethod
    def to_canonical_skeleton(self, skel):
        """Convert output skeleton into a canonical 17-joint skeleton.

        If the dataset is configured to produce canonical skeletons, this method returns
        the skeleton unchanged.

        Args:
            skel: [B x] J x D

        Returns:
            The canonical skeleton, [B x] 17 x D
        """
        pass

    def _evaluate_3d(self, index, original_skel, norm_pred, camera_intrinsics, transform_opts):
        raise NotImplementedError()

    def evaluate_3d_batch(self, batch, norm_preds):
        return [
            self._evaluate_3d(
                batch['index'][i],
                batch['original_skel'][i],
                norm_preds[i],
                batch['camera_intrinsic'][i],
                batch['transform_opts'][i],
            )
            for i in range(len(norm_preds))
            if batch['valid_depth'][i] == 1
        ]

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass


def collate(batch, *, root=True):
    "Puts each data field into a tensor with outer dimension batch size"

    if len(batch) == 0:
        return batch

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if torch.is_tensor(batch[0]):
        return default_collate(batch)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        return default_collate(batch)
    elif isinstance(batch[0], int_classes):
        return batch
    elif isinstance(batch[0], float):
        return batch
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], CameraIntrinsics):
        return batch
    elif isinstance(batch[0], Mapping):
        if root:
            return {key: collate([d[key] for d in batch], root=False) for key in batch[0]}
        else:
            return batch
    elif isinstance(batch[0], Sequence):
        return [collate(e, root=False) for e in batch]

    raise TypeError((error_msg.format(type(batch[0]))))


def worker_init(worker_id):
    seed_all(torch.initial_seed() % 2**32)


def make_dataloader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                    num_workers=0, pin_memory=False, drop_last=False):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
        batch_sampler=batch_sampler, collate_fn=collate, num_workers=num_workers,
        pin_memory=pin_memory, drop_last=drop_last, worker_init_fn=worker_init
    )


class UnbatchedDataLoaderIter(object):
    def __init__(self, loader):
        self.dataset = loader.dataset
        self.sampler = loader.sampler
        self.sample_iter = iter(self.sampler)

    def __len__(self):
        return len(self.sampler)

    def __next__(self):
        index = next(self.sample_iter)  # may raise StopIteration
        return self.dataset[index]

    def __iter__(self):
        return self


class UnbatchedDataLoader(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.sampler = SequentialSampler(dataset)

    def __iter__(self):
        return UnbatchedDataLoaderIter(self)

    def __len__(self):
        return len(self.sampler)


def make_unbatched_dataloader(dataset):
    return UnbatchedDataLoader(dataset)
