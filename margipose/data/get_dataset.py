from margipose.data import PoseDataset
from margipose.data.mpi_inf_3dhp import MpiInf3dDataset
from margipose.data.h36m import H36MDataset
from margipose.data.mpii import MpiiDataset
from margipose.data.mixed import MixedPoseDataset
from margipose.data_specs import DataSpecs


def get_dataset(dataset_name, data_specs=None, use_aug=False) -> PoseDataset:
    """Get a dataset instance by name.

    Args:
        dataset_name (str): The name of the dataset (eg. mpi3d-train)
        data_specs (DataSpecs): The data specs for examples
        use_aug (bool): If true, use data augmentation

    Returns:
        The 3D pose dataset instance.
    """
    if dataset_name == 'mpi3d-train':
        return MpiInf3dDataset('/datasets/mpi3d/train', data_specs=data_specs, use_aug=use_aug)
    elif dataset_name == 'mpi3d-val':
        return MpiInf3dDataset('/datasets/mpi3d/val', data_specs=data_specs, use_aug=use_aug)
    elif dataset_name == 'mpi3d-trainval':
        return MixedPoseDataset([
            get_dataset('mpi3d-train', data_specs, use_aug),
            get_dataset('mpi3d-val', data_specs, use_aug),
        ], balanced_sampling=False)
    elif dataset_name == 'mpi3d-test':
        return MpiInf3dDataset('/datasets/mpi3d/test', data_specs=data_specs, use_aug=False)
    elif dataset_name == 'mpi3d-test-uncorrected':
        return MpiInf3dDataset('/datasets/mpi3d/test-uncorrected', data_specs=data_specs, use_aug=False)
    elif dataset_name == 'h36m-trainval':
        return H36MDataset('/datasets/h36m', data_specs=data_specs, subset='trainval', use_aug=use_aug)
    elif dataset_name == 'h36m-test':
        return H36MDataset('/datasets/h36m', data_specs=data_specs, subset='test', use_aug=use_aug)
    elif dataset_name == 'mpii-train':
        return MpiiDataset('/datasets/mpii', data_specs=data_specs, subset='train', use_aug=use_aug)
    elif dataset_name == 'mpii-val':
        return MpiiDataset('/datasets/mpii', data_specs=data_specs, subset='val', use_aug=use_aug)
    elif dataset_name == 'mpii-trainval':
        return MpiiDataset('/datasets/mpii', data_specs=data_specs, subset='trainval', use_aug=use_aug)
    else:
        raise Exception('unrecognised dataset: {}'.format(dataset_name))
