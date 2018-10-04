import os
from os import path
import numpy as np
from scipy.io import loadmat
import torch
from subprocess import call
from shutil import copy, move
from tqdm import tqdm
from tempfile import TemporaryDirectory
import h5py
import PIL.Image
import PIL.ImageDraw
import PIL.ImageOps
import PIL.ImageFilter
import PIL.ImageChops

from margipose.data.mpi_inf_3dhp.common import Constants, Annotations
from margipose.data.skeleton import absolute_to_root_relative


def _progress(iterator, name):
    return tqdm(iterator, desc='{:10s}'.format(name), ascii=True, leave=False)


def is_image_ok(image_file):
    """Test whether a camera frame image is usable."""

    img = PIL.Image.open(image_file)
    grey = np.array(img).mean(axis=-1)

    # If over 1/3 of the image is white, the flash probably went off in
    # this frame, washing out the image and making it unusable.
    if (grey > 250).sum() > (img.height * img.width) / 3.0:
        return False

    return True


def process_camera_video(in_dir, out_dir, camera_id, frame_indices):
    subdirs = [('imageSequence', 'jpg'), ('ChairMasks', 'png'), ('FGmasks', 'jpg')]

    for subdir, ext in _progress(subdirs, 'Videos'):
        frames_dir = path.join(out_dir, subdir, 'video_%d' % camera_id)
        os.makedirs(frames_dir, exist_ok=True)

        existing_files = {f for f in os.listdir(frames_dir)}
        skip = True
        for i in frame_indices:
            filename = 'img_%06d.%s' % (i + 1, ext)
            if filename not in existing_files:
                skip = False
                break
        if skip:
            continue

        video_file = path.join(in_dir, subdir, 'video_%d.avi' % camera_id)

        with TemporaryDirectory() as tmp_dir:
            call([
                'ffmpeg',
                '-nostats', '-loglevel', '0',
                '-i', video_file,
                '-vf', 'scale=768:768',
                '-qscale:v', '3',
                path.join(tmp_dir, 'img_%06d.{}'.format(ext))
            ])

            for i in frame_indices:
                filename = 'img_%06d.%s' % (i + 1, ext)
                move(
                    path.join(tmp_dir, filename),
                    path.join(frames_dir, filename)
                )


def interesting_frame_indices(annot, camera_id, n_frames):
    """Use the annotations to find interesting training poses.

    A pose is "interesting" if it is sufficiently different from previously seen
    poses, and is within the image bounds.
    """

    univ_annot3 = torch.from_numpy(annot.univ_annot3[camera_id])
    annot2 = torch.from_numpy(annot.annot2[camera_id])

    frame_indices = []
    prev_joints3d = None
    threshold = 200 ** 2  # Require a joint to move at least 200mm since the previous pose
    for i in range(n_frames):
        joints3d = univ_annot3[i]
        if prev_joints3d is not None:
            max_move = (joints3d - prev_joints3d).pow(2).sum(-1).max().item()
            if max_move < threshold:
                continue
        # Keep pose if all joint coordinates are within the image bounds
        if annot2[i].min().item() >= 0 and annot2[i].max().item() < 2048:
            prev_joints3d = joints3d
            frame_indices.append(i)
    return frame_indices


def _add_annotation_metadata(f, annot, n_frames):
    ds = f.create_dataset(
        'joints3d',
        (Constants['n_cameras'], n_frames, 28, 3),
        dtype='f8'
    )
    ds[:] = annot.annot3[:, :n_frames]

    ds = f.create_dataset(
        'scale',
        (1,),
        dtype='f8'
    )
    root_index = Constants['root_joint']
    rel_annot3 = absolute_to_root_relative(torch.from_numpy(annot.annot3), root_index)
    rel_univ = absolute_to_root_relative(torch.from_numpy(annot.univ_annot3), root_index)
    non_zero = rel_univ.abs().gt(1e-6)
    ratio = (rel_annot3 / rel_univ).masked_select(non_zero)
    assert ratio.std().item() < 1e-6
    ds[:] = ratio.mean().item()


def process_sequence(in_dir, out_dir, n_frames, blacklist):
    os.makedirs(out_dir, exist_ok=True)

    for filename in ['annot.mat', 'camera.calibration']:
        src_file = path.join(in_dir, filename)
        dest_file = path.join(out_dir, filename)
        if not path.exists(dest_file):
            copy(src_file, dest_file)

    with h5py.File(path.join(out_dir, 'metadata.h5'), 'w') as f:
        annot = Annotations(loadmat(path.join(out_dir, 'annot.mat')))
        _add_annotation_metadata(f, annot, n_frames)
        for camera_id in _progress(Constants['vnect_cameras'], 'Cameras'):
            if camera_id not in blacklist:
                process_camera_video(in_dir, out_dir, camera_id, range(n_frames))
                indices = interesting_frame_indices(annot, camera_id, n_frames)
                images_dir = path.join(out_dir, 'imageSequence', 'video_%d' % camera_id)
                indices = [
                    i for i in indices
                    if is_image_ok(path.join(images_dir, 'img_%06d.jpg' % (i + 1)))
                ]
                ds = f.create_dataset(
                    'interesting_frames/camera%d' % camera_id,
                    (len(indices),),
                    dtype='i8'
                )
                ds[:] = np.array(indices)


def preprocess_sequences(src_dir, dest_dir, seqs):
    for subj_id, seq_id in _progress(seqs, 'Sequences'):
        seq_rel_path = path.join('S%d' % subj_id, 'Seq%d' % seq_id)
        process_sequence(
            path.join(src_dir, seq_rel_path),
            path.join(dest_dir, seq_rel_path),
            n_frames=Constants['seq_info'][seq_rel_path]['num_frames'],
            blacklist=Constants['blacklist'].get(seq_rel_path, [])
        )


def preprocess_training_data(src_dir, dest_dir):
    return preprocess_sequences(src_dir, dest_dir, Constants['train_seqs'])


def preprocess_validation_data(src_dir, dest_dir):
    return preprocess_sequences(src_dir, dest_dir, Constants['val_seqs'])


def preprocess_test_data(src_dir, dest_dir):
    from margipose.data.mpi_inf_3dhp.raw import RawMpiTestDataset, RawMpiTestSeqDataset

    for seq_id in _progress(RawMpiTestDataset.SEQ_IDS, 'Sequences'):
        dataset = RawMpiTestSeqDataset(src_dir, seq_id, valid_only=True)

        out_dir = path.join(dest_dir, seq_id.replace('TS', 'S'), 'Seq1')
        image_out_dir = path.join(out_dir, 'imageSequence', 'video_0')
        os.makedirs(image_out_dir, exist_ok=True)

        image_width = image_height = -1
        for example in _progress(dataset, 'Images'):
            image = PIL.Image.open(example['image_file'])
            image_width, image_height = image.size
            image = image.resize((int(image_width * 768 / image_height), 768), PIL.Image.ANTIALIAS)
            image.save(path.join(image_out_dir, 'img_%06d.jpg' % (example['frame_index'] + 1)))

        copy(dataset.annot_file, path.join(out_dir, 'annot_data.mat'))

        with h5py.File(path.join(out_dir, 'metadata.h5'), 'w') as f:
            with h5py.File(dataset.annot_file, 'r') as annot:
                n_frames = len(annot['annot3'])
                annot3 = np.array(annot['annot3']).reshape(1, n_frames, 17, 3)
                univ_annot3 = np.array(annot['univ_annot3']).reshape(1, n_frames, 17, 3)
                annot2 = np.array(annot['annot2']).reshape(1, n_frames, 17, 2)

                # Infer camera intrinsics
                x3d = np.stack([annot3[0, :, :, 0], annot3[0, :, :, 2]], axis=-1).reshape(n_frames * 17, 2)
                x2d = (annot2[0, :, :, 0] * annot3[0, :, :, 2]).reshape(n_frames * 17, 1)
                fx, cx = list(np.linalg.lstsq(x3d, x2d, rcond=None)[0].flatten())
                y3d = np.stack([annot3[0, :, :, 1], annot3[0, :, :, 2]], axis=-1).reshape(n_frames * 17, 2)
                y2d = (annot2[0, :, :, 1] * annot3[0, :, :, 2]).reshape(n_frames * 17, 1)
                fy, cy = list(np.linalg.lstsq(y3d, y2d, rcond=None)[0].flatten())

                with open(path.join(out_dir, 'camera.calibration'), 'w') as cam_file:
                    lines = [
                        'Fake Camera Calibration File',
                        'name          0',
                        '  size        {:d} {:d}'.format(image_width, image_height),
                        '  intrinsic   {:0.3f} 0 {:0.3f} 0 0 {:0.3f} {:0.3f} 0 0 0 1 0 0 0 0 1'
                            .format(fx, cx, fy, cy),
                        '  extrinsic   1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1',
                    ]
                    for line in lines:
                        cam_file.write(line + '\n')

                ds = f.create_dataset('joints3d', (1, n_frames, 17, 3), dtype='f8')
                ds[:] = np.array(annot3).reshape(1, n_frames, 17, 3)

                root_index = Constants['root_joint']
                rel_annot3 = absolute_to_root_relative(torch.from_numpy(annot3), root_index)
                rel_univ = absolute_to_root_relative(torch.from_numpy(univ_annot3), root_index)
                non_zero = rel_univ.abs().gt(1e-6)
                ratio = (rel_annot3 / rel_univ).masked_select(non_zero)
                assert ratio.std() < 1e-6
                ds = f.create_dataset('scale', (1,), dtype='f8')
                ds[:] = ratio.mean()

                indices = []
                for frame_index, is_valid in enumerate(np.array(annot['valid_frame']).flatten()):
                    if is_valid == 1:
                        indices.append(frame_index)
                ds = f.create_dataset( 'interesting_frames/camera0', (len(indices),), dtype='i8')
                ds[:] = np.array(indices)


def _isolate_person(img, skel2d):
    x1, y1 = list(skel2d.min(axis=0))
    x2, y2 = list(skel2d.max(axis=0))
    margin = 30
    x1 = max(x1 - margin, 0)
    y1 = max(y1 - margin, 0)
    x2 = min(x2 + margin, 767)
    y2 = min(y2 + margin, 767)

    draw = PIL.ImageDraw.Draw(img)
    draw.rectangle([0, 0, x1, 767], fill=0)
    draw.rectangle([0, 0, 767, y1], fill=0)
    draw.rectangle([x2, 0, 767, 767], fill=0)
    draw.rectangle([0, y2, 767, 767], fill=0)


def preprocess_masks(dir, subj_id, seq_id):
    # Masks are useful to do data augmentation with compositing
    # eg `PIL.Image.composite(example_input, shirt_pattern, up_body_mask)`

    seq_rel_path = path.join('S%d' % subj_id, 'Seq%d' % seq_id)
    seq_dir = path.join(dir, seq_rel_path)

    info = Constants['seq_info'][seq_rel_path]

    interesting_frames = []
    with h5py.File(path.join(seq_dir, 'metadata.h5'), 'r') as f:
        for k in f['interesting_frames'].keys():
            interesting_frames.append(
                (int(k.replace('camera', '')), list(f['interesting_frames'][k]))
            )

    annot = Annotations(loadmat(path.join(seq_dir, 'annot.mat')))

    for camera_id, frame_indices in _progress(interesting_frames, 'Cameras'):
        for frame_index in frame_indices:
            path_part = 'video_{}/img_{:06d}'.format(camera_id, frame_index + 1)

            img = PIL.Image.open(path.join(seq_dir, 'FGmasks/{}.jpg'.format(path_part)))
            img = PIL.ImageOps.invert(img)
            fg, up_body, low_body = img.split()

            skel2d = annot.annot2[camera_id, frame_index] * 768 / 2048

            if info['bg_augmentable']:
                fg = PIL.ImageOps.invert(fg)
                _isolate_person(fg, skel2d)

                chair = PIL.Image.open(
                    path.join(seq_dir, 'ChairMasks/{}.png'.format(path_part)))
                chair, _, _ = chair.split()
                chair = PIL.ImageOps.invert(chair)

                # Pixel-wise max
                combined = PIL.ImageChops.lighter(fg, chair)

                fg_file = path.join(seq_dir, 'foreground_mask', path_part + '.png')
                os.makedirs(path.dirname(fg_file), exist_ok=True)
                combined.save(fg_file)

            if info['ub_augmentable']:
                _isolate_person(up_body, skel2d)
                up_body = up_body.filter(PIL.ImageFilter.MinFilter(3))
                up_body = up_body.filter(PIL.ImageFilter.MaxFilter(3))

                up_body_file = path.join(seq_dir, 'up_body_mask', path_part + '.png')
                os.makedirs(path.dirname(up_body_file), exist_ok=True)
                up_body.save(up_body_file)

            if info['lb_augmentable']:
                _isolate_person(low_body, skel2d)
                low_body = low_body.filter(PIL.ImageFilter.MinFilter(3))
                low_body = low_body.filter(PIL.ImageFilter.MaxFilter(3))

                low_body_file = path.join(seq_dir, 'low_body_mask', path_part + '.png')
                os.makedirs(path.dirname(low_body_file), exist_ok=True)
                low_body.save(low_body_file)


def preprocess_training_masks(dir):
    """Preprocess masks in a preprocessed training data directory."""

    for subj_id, seq_id in _progress(Constants['train_seqs'], 'Sequences'):
        preprocess_masks(dir, subj_id, seq_id)


def preprocess_validation_masks(dir):
    """Preprocess masks in a preprocessed validation data directory."""

    for subj_id, seq_id in _progress(Constants['val_seqs'], 'Sequences'):
        preprocess_masks(dir, subj_id, seq_id)
