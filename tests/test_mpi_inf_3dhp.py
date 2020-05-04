import os
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
