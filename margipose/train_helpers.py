"""Helper code for training human pose estimation networks."""

from os import environ
import torch
from torch import optim
from tqdm import tqdm
import pyshowoff

from margipose.data import PoseDataset, make_dataloader
from margipose.data.mixed import MixedPoseDataset
from margipose.data.get_dataset import get_dataset
from margipose.utils import draw_skeleton_2d


def visualise_predictions(preds, batch, dataset):
    """Create images with overlaid predictions (for visualisation purposes).

    Args:
        preds: The predicted skeletons.
        batch (dict): The sample batch corresponding to the predictions.
        dataset (PoseDataset): The dataset from which the sample batch originates.

    Returns:
        A list of images with predicted poses overlaid.
    """
    if preds.size(-1) < 4:
        preds = torch.cat([preds, torch.ones_like(preds.narrow(-1, 0, 4 - preds.size(-1)))], -1)
    images = []
    for i in range(len(batch['input'])):
        img = dataset.input_to_pil_image(batch['input'][i])
        camera_intrinsics = batch['camera_intrinsic'][i]
        skel = dataset.to_image_space(batch['index'][i], preds[i], camera_intrinsics)
        draw_skeleton_2d(img, skel, dataset.skeleton_desc)
        images.append(img)
    return images


def progress_iter(iterable, name):
    return tqdm(iterable, desc='{:10s}'.format(name), leave=True, ascii=True)


def create_showoff_notebook(title, extra_tags) -> pyshowoff.Notebook:
    client = pyshowoff.Client(
        environ.get('SHOWOFF_URL'),
        environ.get('SHOWOFF_KEY_ID'),
        environ.get('SHOWOFF_SECRET_KEY'),
    )
    notebook: pyshowoff.Notebook = client.add_notebook(title).result()
    with open('/etc/hostname', 'r') as f:
        hostname = f.read().strip()
    tags = [hostname] + extra_tags
    for tag in tags:
        notebook.add_tag(tag).result()
    return notebook


def learning_schedule(params, optim_algorithm, lr, milestones, gamma):
    """Creates an optimizer and learning rate scheduler.

    Args:
        params: Model parameters
        optim_algorithm (str): Name of the optimisation algorithm
        lr: Initial learning rate
        milestones: Schedule milestones
        gamma: Learning rate decay factor

    Returns:
        optim.lr_scheduler._LRScheduler: Learning rate scheduler
    """
    if optim_algorithm == 'sgd':
        optimiser = optim.SGD(params, lr=lr)
    elif optim_algorithm == 'nesterov':
        optimiser = optim.SGD(params, lr=lr, momentum=0.8, nesterov=True)
    elif optim_algorithm == 'rmsprop':
        optimiser = optim.RMSprop(params, lr=lr)
    else:
        raise Exception('unrecognised optimisation algorithm: ' + optim_algorithm)
    return optim.lr_scheduler.MultiStepLR(optimiser, milestones=milestones, gamma=gamma)


def _create_dataloader(dataset_names, data_specs, batch_size, examples_per_epoch, use_aug):
    datasets = [get_dataset(name, data_specs, use_aug=use_aug) for name in dataset_names]
    assert len(datasets) > 0, 'at least one dataset must be specified'
    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = MixedPoseDataset(datasets)
    return make_dataloader(
        dataset,
        sampler=dataset.sampler(examples_per_epoch=examples_per_epoch),
        batch_size=batch_size,
        drop_last=True,
        num_workers=4,
    )


def create_train_dataloader(dataset_names, data_specs, batch_size, examples_per_epoch):
    return _create_dataloader(dataset_names, data_specs, batch_size, examples_per_epoch,
                              use_aug=True)


def create_val_dataloader(dataset_names, data_specs, batch_size, examples_per_epoch):
    return _create_dataloader(dataset_names, data_specs, batch_size, examples_per_epoch,
                              use_aug=False)
