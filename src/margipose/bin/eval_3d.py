#!/usr/bin/env python3

"""Calculate evaluation metrics for a trained model."""


import argparse
import json

import torch
from pose3d_utils.coords import ensure_homogeneous
from tele.meter import MeanValueMeter, MedianValueMeter
from tqdm import tqdm

from margipose.data import make_dataloader, make_unbatched_dataloader
from margipose.data.get_dataset import get_dataset
from margipose.data.skeleton import CanonicalSkeletonDesc, VNect_Common_Skeleton
from margipose.dsntnn import average_loss
from margipose.eval import prepare_for_3d_evaluation, gather_3d_metrics
from margipose.models import load_model
from margipose.utils import seed_all, init_algorithms
from margipose.utils import timer

CPU = torch.device('cpu')
GPU = torch.device('cuda')


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description='3D human pose model evaluator')
    parser.add_argument('--model', type=str, metavar='FILE', required=True,
                        help='path to model file')
    parser.add_argument('--dataset', type=str, metavar='DS', default='mpi3d-test',
                        help='dataset to evaluate on')
    parser.add_argument('--multicrop', action='store_true',
                        help='enable the use of multiple crops')

    args = parser.parse_args()

    return args


def run_evaluation_3d(model, loader, included_joints, known_depth=False, print_progress=False):
    loss_meter = MeanValueMeter()
    time_meter = MedianValueMeter()

    metrics = []

    model.eval()

    iterable = loader
    if print_progress:
        iterable = tqdm(loader, leave=True, ascii=True)

    for batch in iterable:
        in_var = batch['input'].to(GPU, torch.float32)
        target_var = batch['target'].to(GPU, torch.float32)

        # Calculate predictions and loss
        with timer(time_meter):
            out_var = model(in_var)
        loss = average_loss(model.forward_3d_losses(out_var, target_var.narrow(-1, 0, 3)))

        loss_meter.add(loss.sum().item())

        norm_preds = ensure_homogeneous(out_var.to(CPU, torch.float64), d=3)

        actuals = []
        expected = None
        for i, norm_pred in enumerate(norm_preds):
            expected_i, actual_i =\
                prepare_for_3d_evaluation(batch['original_skel'][i], norm_pred,
                                          loader.dataset, batch['camera_intrinsic'][i],
                                          batch['transform_opts'][i], known_depth=known_depth)
            if expected is not None:
                assert (expected_i - expected).abs().gt(1e-6).sum() == 0,\
                    "Expected all examples in batch to have the same target"
            expected = expected_i
            actuals.append(actual_i)
        actual = torch.stack(actuals, 0).mean(0)

        metrics.append(gather_3d_metrics(expected, actual, included_joints))

    aggregated_metrics = dict(
        loss=loss_meter.value()[0],
        time=time_meter.value()[0],
    )

    # Aggregate the mean value of each 3D evaluation metric
    for metric_name in metrics[0].keys():
        metric_values = [m[metric_name] for m in metrics]
        aggregated_metrics[metric_name] = sum(metric_values) / len(metric_values)

    return aggregated_metrics


def main():
    args = parse_args()
    seed_all(12345)
    init_algorithms(deterministic=True)
    torch.set_grad_enabled(False)

    model = load_model(args.model).to(GPU).eval()
    dataset = get_dataset(args.dataset, model.data_specs, use_aug=False)

    if args.multicrop:
        dataset.multicrop = True
        loader = make_unbatched_dataloader(dataset)
    else:
        loader = make_dataloader(dataset, batch_size=1)

    if args.dataset.startswith('h36m-'):
        known_depth = True
        included_joints = list(range(CanonicalSkeletonDesc.n_joints))
    else:
        known_depth = False
        included_joints = [
            CanonicalSkeletonDesc.joint_names.index(joint_name)
            for joint_name in VNect_Common_Skeleton
        ]
    print('Use ground truth root joint depth? {}'.format(known_depth))
    print('Number of joints in evaluation: {}'.format(len(included_joints)))

    metrics = run_evaluation_3d(model, loader, included_joints, known_depth=known_depth, print_progress=True)

    print(json.dumps(metrics, sort_keys=True, indent=2))
    print(','.join([
        '{:0.6f}%'.format(metrics['pck'] * 100),
        '{:0.6f}'.format(metrics['mpjpe']),
        '{:0.6f}%'.format(metrics['auc'] * 100),
        '{:0.6f}'.format(1.0 / metrics['time']),
        '',
        '{:0.6f}%'.format(metrics['aligned_pck'] * 100),
        '{:0.6f}'.format(metrics['aligned_mpjpe']),
        '{:0.6f}%'.format(metrics['aligned_auc'] * 100),
    ]))


if __name__ == '__main__':
    main()
