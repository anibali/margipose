import os
import pytest
from torch.testing import assert_allclose

from margipose.data.mpi_inf_3dhp import MpiInf3dDataset
from margipose.data.skeleton import CanonicalSkeletonDesc
from margipose.data_specs import DataSpecs, ImageSpecs, JointsSpecs


def test_to_canonical_skeleton(skeleton_mpi3d_univ, skeleton_canonical_univ):
    actual = MpiInf3dDataset._mpi_inf_3dhp_to_canonical_skeleton(skeleton_mpi3d_univ)
    assert_allclose(actual, skeleton_canonical_univ)


def test_mpi3d_val_subset(mpi3d_data_dir):
    data_specs = DataSpecs(
        ImageSpecs(256, mean=ImageSpecs.IMAGENET_MEAN, stddev=ImageSpecs.IMAGENET_STDDEV),
        JointsSpecs(CanonicalSkeletonDesc, n_dims=3),
    )
    dataset = MpiInf3dDataset(os.path.join(mpi3d_data_dir, 'val'), data_specs)
    assert len(dataset) == 18561
    example = dataset[0]
    assert example['input'].shape == (3, 256, 256)


def test_mpi3d_example_data(mpi3d_data_dir):
    data_specs = DataSpecs(
        ImageSpecs(256, mean=ImageSpecs.IMAGENET_MEAN, stddev=ImageSpecs.IMAGENET_STDDEV),
        JointsSpecs(CanonicalSkeletonDesc, n_dims=3),
    )
    dataset = MpiInf3dDataset(os.path.join(mpi3d_data_dir, 'val'), data_specs)
    assert MpiInf3dDataset.preserve_root_joint_at_univ_scale == False
    example = dataset[0]

    image = example['input']
    assert float(image.min()) == pytest.approx(-2.117904, rel=0, abs=1e-2)
    assert float(image.max()) == pytest.approx(2.428571, rel=0, abs=1e-2)

    joints = example['target'][..., :3]
    assert_allclose(joints[0], [-0.025768, -0.649297, -0.039933], rtol=0, atol=1e-4)
