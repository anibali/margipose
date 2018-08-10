import json
import re

import numpy as np
import torch
from importlib_resources import read_text
from pose3d_utils.camera import CameraIntrinsics

from margipose.data.skeleton import SkeletonDesc

Constants = {
    # Training set sequences: (subject_id, sequence_id)
    'train_seqs': [
        (1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (4, 2),
        (5, 1), (5, 2), (6, 1), (6, 2), (7, 1), (7, 2), (8, 1),
    ],
    # Validation set sequences: (subject_id, sequence_id)
    'val_seqs': [
        (4, 1), (8, 2)
    ],
    # Camera IDs used for training/validation sets (same as VNect)
    'vnect_cameras': [0, 1, 2, 4, 5, 6, 7, 8],
    # Total number of cameras
    'n_cameras': 14,
    # Per-sequence information extracted from "mpii_get_sequence_info.m"
    'seq_info': json.loads(read_text('margipose.data.mpi_inf_3dhp', 'sequence_info.json')),
    # Root joint index (pelvis)
    'root_joint': 4,
    # Videos with known problems
    'blacklist': {
        'S6/Seq2': [2],  # imageSequence/video_2.avi is too short
    },
}

MpiInf3dhpSkeletonDesc = SkeletonDesc(
    joint_names=[
        # 0-3
        'spine3', 'spine4', 'spine2', 'spine',
        # 4-7
        'pelvis', 'neck', 'head', 'head_top',
        # 8-11
        'left_clavicle', 'left_shoulder', 'left_elbow', 'left_wrist',
        # 12-15
        'left_hand',  'right_clavicle', 'right_shoulder', 'right_elbow',
        # 16-19
        'right_wrist', 'right_hand', 'left_hip', 'left_knee',
        # 20-23
        'left_ankle', 'left_foot', 'left_toe', 'right_hip',
        # 24-27
        'right_knee', 'right_ankle', 'right_foot', 'right_toe'
    ],
    joint_tree=[
        2, 0, 3, 4,
        4, 1, 5, 6,
        5, 8, 9, 10,
        11, 5, 13, 14,
        15, 16, 4, 18,
        19, 20, 21, 4,
        23, 24, 25, 26
    ],
    hflip_indices=[
        0, 1, 2, 3,
        4, 5, 6, 7,
        13, 14, 15, 16,
        17, 8, 9, 10,
        11, 12, 23, 24,
        25, 26, 27, 18,
        19, 20, 21, 22
    ]
)


class Annotations:
    def __init__(self, annot):
        self.annot = annot
        assert np.array_equal(annot['cameras'].flatten(), np.arange(Constants['n_cameras']))
        assert np.array_equal(annot['frames'].flatten(), np.arange(annot['frames'].shape[0]))
        self.annot3 = self._reshape_annot(annot['annot3'], 3)
        self.univ_annot3 = self._reshape_annot(annot['univ_annot3'], 3)
        self.annot2 = self._reshape_annot(annot['annot2'], 2)

    @staticmethod
    def _reshape_annot(arr, ndims):
        arr = np.stack(arr.flatten())
        return arr.reshape((arr.shape[0], arr.shape[1], 28, ndims))


def parse_camera_calibration(f):
    line_re = re.compile(r'(\w+)\s+(.+)')
    types = {
        'name': 'int',
        'sensor': 'vec2',
        'size': 'vec2',
        'animated': 'int',
        'intrinsic': 'mat4',
        'extrinsic': 'mat4',
        'radial': 'int',
    }
    f.readline()
    camera_properties = {}
    props = None
    for line in f.readlines():
        line_match = line_re.fullmatch(line.strip())
        if line_match:
            key, value = line_match.groups()
            values = value.split(' ')
            value_type = types[key]
            if value_type == 'int':
                assert len(values) == 1
                parsed_value = int(values[0])
            elif value_type == 'vec2':
                assert len(values) == 2
                parsed_value = np.array([float(v) for v in values])
            elif value_type == 'mat4':
                assert len(values) == 4 * 4
                parsed_value = np.array([float(v) for v in values]).reshape((4, 4))
            else:
                print('Skipping unrecognized camera calibration field:', key)
                continue

            if key == 'name':
                props = {}
                camera_properties[parsed_value] = props
            else:
                props[key] = parsed_value

    cameras = {}
    for i, props in camera_properties.items():
        cameras[i] = {
            'intrinsics': CameraIntrinsics(torch.from_numpy(props['intrinsic'])[:3]),
            'extrinsics': torch.from_numpy(props['extrinsic']),
            'image_width': props['size'][0],
            'image_height': props['size'][1],
        }

    return cameras
