from torch.testing import assert_allclose

from margipose.data.mpi_inf_3dhp import MpiInf3dDataset


def test_to_canonical_skeleton(skeleton_mpi3d_univ, skeleton_canonical_univ):
    actual = MpiInf3dDataset._mpi_inf_3dhp_to_canonical_skeleton(skeleton_mpi3d_univ)
    assert_allclose(actual, skeleton_canonical_univ)
