import random
from torch.utils.data.sampler import Sampler
from margipose.data import PoseDataset


class RoundRobinSampler(Sampler):
    def __init__(self, index_lists, num_samples):
        super().__init__(None)
        self.index_lists = index_lists
        self.num_samples = num_samples

    def __iter__(self):
        shuffled_index_lists = [list(index_list) for index_list in self.index_lists]
        for l in shuffled_index_lists:
            random.shuffle(l)
        i = 0
        js = [0] * len(shuffled_index_lists)
        for _ in range(len(self)):
            yield shuffled_index_lists[i][js[i]]
            js[i] += 1
            i = (i + 1) % len(js)

    def __len__(self):
        return self.num_samples


class MixedPoseDataset(PoseDataset):
    """Multiple pose datasets combined into one.

    Args:
        datasets (list of PoseDataset):
    """

    def __init__(self, datasets, balanced_sampling=True):
        # Enforce same data specs for all datasets
        data_specs = datasets[0].data_specs
        for dataset in datasets[1:]:
            other_data_specs = dataset.data_specs
            assert other_data_specs == data_specs, 'combined datasets must have same data specs'

        super().__init__(data_specs)

        self.datasets = datasets
        self.dataset_lengths = [len(d) for d in datasets]
        self.length = sum(self.dataset_lengths)
        self.balanced_sampling = balanced_sampling

        self.per_dataset_indices = [[] for _ in datasets]
        for i in range(len(self)):
            dataset_index, _ = self._decompose_index(i)
            self.per_dataset_indices[dataset_index].append(i)

    def _decompose_index(self, index):
        upper = 0
        for i, length in enumerate(self.dataset_lengths):
            offset = upper
            upper += length
            if index < upper:
                return i, index - offset
        raise Exception('index out of bounds')

    def sampler(self, examples_per_epoch=None):
        # Use the normal sampler if balanced_sampling is False.
        if not self.balanced_sampling:
            return super().sampler(examples_per_epoch)
        return RoundRobinSampler(self.per_dataset_indices, examples_per_epoch)

    def _evaluate_3d(self, index, original_skel, norm_pred, camera_intrinsics, transform_opts):
        dataset_index, example_index = self._decompose_index(index)
        return self.datasets[dataset_index]._evaluate_3d(example_index, original_skel, norm_pred,
                                                         camera_intrinsics, transform_opts['opts'])

    def to_image_space(self, index, normalised, intrinsics):
        dataset_index, example_index = self._decompose_index(index)
        dataset = self.datasets[dataset_index]
        return dataset.to_image_space(example_index, normalised, intrinsics)

    def untransform_skeleton(self, denorm_skel, trans_opts):
        dataset_index = trans_opts['dataset_index']
        return self.datasets[dataset_index].untransform_skeleton(denorm_skel, trans_opts['opts'])

    def to_canonical_skeleton(self, skel):
        return self.datasets[0].to_canonical_skeleton(skel)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        dataset_index, example_index = self._decompose_index(index)
        example = self.datasets[dataset_index][example_index]

        sample = {
            'index': index,
            'valid_depth': example['valid_depth'],

            'original_skel': example['original_skel'],

            'input': example['input'],
            'camera_intrinsic': example['camera_intrinsic'],
            'camera_extrinsic': example['camera_extrinsic'],
            'target': example['target'],
            'joint_mask': example['joint_mask'],

            'transform_opts': {
                'dataset_index': dataset_index,
                'opts': example['transform_opts'],
            },
        }

        return sample
