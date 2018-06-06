"""
Code for loading raw (unprocessed) MPI-INF-3DHP data.
"""

from os import path
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np


class RawMpiTestSeqDataset(Dataset):
    def __init__(self, data_dir, seq_id, valid_only=True):
        frame_indices = []

        with h5py.File(path.join(data_dir, seq_id, 'annot_data.mat'), 'r') as annot:
            if valid_only:
                new_frame_indices = list(np.where(annot['valid_frame'])[0])
            else:
                new_frame_indices = list(range(len(annot['valid_frame'])))
        frame_indices += new_frame_indices

        self.data_dir = data_dir
        self.frame_indices = frame_indices
        self.seq_id = seq_id
        self.annot_file = path.join(self.data_dir, self.seq_id, 'annot_data.mat')

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, index):
        frame_index = self.frame_indices[index]
        image_id = frame_index + 1
        image_file = path.join(
            self.data_dir, self.seq_id, 'imageSequence', 'img_%06d.jpg' % image_id)

        with h5py.File(self.annot_file, 'r') as annot:
            return {
                'image_file': image_file,
                'seq_id': self.seq_id,
                'frame_index': frame_index,
                'valid': int(annot['valid_frame'][frame_index][0]),
                'annot2': torch.from_numpy(annot['annot2'][frame_index][0]),
                'annot3': torch.from_numpy(annot['annot3'][frame_index][0]),
                'univ_annot3': torch.from_numpy(annot['univ_annot3'][frame_index][0]),
            }


class RawMpiTestDataset(Dataset):
    # Names of test sequences
    SEQ_IDS = ['TS1', 'TS2', 'TS3', 'TS4', 'TS5', 'TS6']

    def __init__(self, data_dir, valid_only=True):
        self.seq_datasets = [
            RawMpiTestSeqDataset(data_dir, seq_id, valid_only=valid_only)
            for seq_id in self.SEQ_IDS
        ]

        seq_indices = []
        frame_indices = []
        seq_start_indices = {}
        for seq_index, seq_dataset in enumerate(self.seq_datasets):
            seq_start_indices[seq_dataset.seq_id] = len(frame_indices)
            frame_indices += list(range(len(seq_dataset)))
            seq_indices += [seq_index] * len(seq_dataset)

        self.data_dir = data_dir
        self.frame_indices = frame_indices
        self.seq_indices = seq_indices
        self.seq_start_indices = seq_start_indices

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, index):
        return self.seq_datasets[self.seq_indices[index]][self.frame_indices[index]]
