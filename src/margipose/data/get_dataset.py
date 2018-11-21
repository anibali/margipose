from os import path, environ
import re

from margipose.data import PoseDataset
from margipose.data.mpi_inf_3dhp import MpiInf3dDataset
from margipose.data.h36m import H36MDataset
from margipose.data.mpii import MpiiDataset
from margipose.data.mixed import MixedPoseDataset
from margipose.data_specs import DataSpecs


# A custom base data directory can be set using the MARGIPOSE_BASE_DATA_DIR environment variable,
# which is useful when running `margipose` outside of a Docker container.
Base_Data_Dir = environ.get('MARGIPOSE_BASE_DATA_DIR', '/datasets')


def get_dataset(dataset_name, data_specs=None, use_aug=False) -> PoseDataset:
    """Get a dataset instance by name.

    Args:
        dataset_name (str): The name of the dataset (eg. mpi3d-train)
        data_specs (DataSpecs): The data specs for examples
        use_aug (bool): If true, use data augmentation

    Returns:
        The pose dataset instance.
    """

    # MPI-INF-3DHP
    mpi3d_match = re.fullmatch('mpi3d-(train|val|test|test-uncorrected)', dataset_name)
    if mpi3d_match:
        subset = mpi3d_match[1]
        return MpiInf3dDataset(path.join(Base_Data_Dir, 'mpi3d', subset),
                               data_specs=data_specs,
                               use_aug=(use_aug and not subset.startswith('test')))
    if dataset_name == 'mpi3d-trainval':
        return MixedPoseDataset([
            get_dataset('mpi3d-train', data_specs, use_aug),
            get_dataset('mpi3d-val', data_specs, use_aug),
        ], balanced_sampling=False)

    # Human3.6M
    h36m_match = re.match('h36m-(trainval|test)', dataset_name)
    if h36m_match:
        subset = h36m_match[1]
        return H36MDataset(path.join(Base_Data_Dir, 'h36m'),
                           data_specs=data_specs, subset=subset,
                           use_aug=(use_aug and subset != 'test'))

    # MPII Human Pose (2D)
    mpii_match = re.match('mpii-(train|val|trainval|test)', dataset_name)
    if mpii_match:
        subset = mpii_match[1]
        return MpiiDataset(path.join(Base_Data_Dir, 'mpii'),
                           data_specs=data_specs, subset=subset,
                           use_aug=(use_aug and subset != 'test'))

    raise Exception('unrecognised dataset: {}'.format(dataset_name))
