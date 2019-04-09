#!/usr/bin/env python3

"""Calculate evaluation metrics for a trained model."""


import argparse
from time import perf_counter

import torch
import pandas as pd
from pose3d_utils.coords import ensure_homogeneous
from tele.meter import MeanValueMeter, MedianValueMeter
from tqdm import tqdm
from tabulate import tabulate

from margipose.cli import Subcommand
from margipose.data import make_dataloader, make_unbatched_dataloader
from margipose.data.get_dataset import get_dataset
from margipose.data.skeleton import CanonicalSkeletonDesc, VNect_Common_Skeleton
from margipose.dsntnn import average_loss
from margipose.eval import prepare_for_3d_evaluation, gather_3d_metrics
from margipose.models import load_model
from margipose.utils import seed_all, init_algorithms


CPU = torch.device('cpu')


def parse_args(argv):
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(prog='margipose-eval',
                                     description='3D human pose model evaluator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, metavar='FILE', default=argparse.SUPPRESS,
                        required=True,
                        help='path to model file')
    parser.add_argument('--dataset', type=str, metavar='DS', default='mpi3d-test',
                        help='dataset to evaluate on')
    parser.add_argument('--multicrop', action='store_true',
                        help='enable the use of multiple crops')

    args = parser.parse_args(argv[1:])

    return args


def obtain_predictions(model, device, loader, known_depth=False, print_progress=False):
    model.eval()

    iterable = loader
    if print_progress:
        iterable = tqdm(loader, leave=True, ascii=True)

    for batch in iterable:
        in_var = batch['input'].to(device, torch.float32)
        target_var = batch['target'].to(device, torch.float32)

        # Calculate predictions and loss
        start_time = perf_counter()
        out_var = model(in_var)
        inference_time = perf_counter() - start_time
        loss = average_loss(model.forward_3d_losses(out_var, target_var.narrow(-1, 0, 3)))

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

        try:
            frame_ref = batch['frame_ref'][0]
        except KeyError:
            frame_ref = None

        prediction = dict(
            expected=expected,
            actual=actual,
            frame_ref=frame_ref,
            inference_time=inference_time,
            loss=loss.sum().item(),
        )

        yield prediction


def run_evaluation_3d(model, device, loader, included_joints, known_depth=False,
                      print_progress=False):
    loss_meter = MeanValueMeter()
    time_meter = MedianValueMeter()

    d = dict(seq_id=[], activity_id=[], aligned_auc=[], aligned_mpjpe=[], aligned_pck=[],
             auc=[], mpjpe=[], pck=[])

    for pred in obtain_predictions(model, device, loader, known_depth, print_progress):
        time_meter.add(pred['inference_time'])
        loss_meter.add(pred['loss'])
        metrics = gather_3d_metrics(pred['expected'], pred['actual'], included_joints)
        if pred['frame_ref']:
            d['seq_id'].append(f'TS{pred["frame_ref"]["subject_id"]}/Seq{pred["frame_ref"]["sequence_id"]}')
            d['activity_id'].append(pred['frame_ref']['activity_id'])
        else:
            d['seq_id'].append('-')
            d['activity_id'].append('-')
        for metric_name, metric_value in metrics.items():
            d[metric_name].append(metric_value)

    return pd.DataFrame(d)


def main(argv, common_opts):
    args = parse_args(argv)
    seed_all(12345)
    init_algorithms(deterministic=True)
    torch.set_grad_enabled(False)

    device = common_opts['device']

    model = load_model(args.model).to(device).eval()
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

    df = run_evaluation_3d(model, device, loader, included_joints, known_depth=known_depth,
                           print_progress=True)

    print('### By sequence')
    print()
    print(tabulate(df.drop(columns=['activity_id']).groupby('seq_id').mean(), headers='keys', tablefmt='pipe'))
    print()
    print('### By activity')
    print()
    print(tabulate(df.drop(columns=['seq_id']).groupby('activity_id').mean(), headers='keys', tablefmt='pipe'))
    print()
    print('### Overall')
    print()
    print(tabulate(df.drop(columns=['activity_id', 'seq_id']).mean().to_frame().T, headers='keys', tablefmt='pipe'))


Eval_Subcommand = Subcommand(name='eval', func=main, help='evaluate the accuracy of predictions')

if __name__ == '__main__':
    Eval_Subcommand.run()
