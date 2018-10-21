#!/usr/bin/env python3

"""Search for good training hyperparameters.

This code runs the LR range test proposed in "Cyclical Learning Rates for Training Neural Networks"
by Leslie N. Smith.
"""

import json
from os import path, environ

import numpy as np
import plotly.graph_objs as go
import sacred
import tele
import torch
from sacred.host_info import get_host_info
from sacred.run import Run
from tele.meter import ValueMeter
from torch.optim import SGD
from tqdm import tqdm

from margipose.cli import Subcommand
from margipose.dsntnn import average_loss
from margipose.models import create_model
from margipose.models.margipose_model import Default_MargiPose_Desc
from margipose.models.chatterbox_model import Default_Chatterbox_Desc
from margipose.train_helpers import create_train_dataloader, create_showoff_notebook
from margipose.utils import seed_all, init_algorithms

sacred.SETTINGS['DISCOVER_SOURCES'] = 'dir'
ex = sacred.Experiment(base_dir=path.realpath(path.join(__file__, '..', '..')))

global_opts = {}


def forward_loss(model, out_var, target_var, mask_var, valid_depth):
    target_var = target_var.narrow(-1, 0, 3)

    if not 0 in valid_depth:
        losses = model.forward_3d_losses(out_var, target_var)
    elif not 1 in valid_depth:
        losses = model.forward_2d_losses(out_var, target_var)
    else:
        losses_3d = model.forward_3d_losses(out_var, target_var)
        losses_2d = model.forward_2d_losses(out_var, target_var)
        losses = torch.stack([
            (losses_3d[i] if use_3d == 1 else losses_2d[i])
            for i, use_3d in enumerate(valid_depth)
        ])

    return average_loss(losses, mask_var)


ex.add_named_config('margipose_model', model_desc=Default_MargiPose_Desc)
ex.add_named_config('chatterbox_model', model_desc=Default_Chatterbox_Desc)

ex.add_config(
    showoff=not not environ.get('SHOWOFF_URL'),
    batch_size=32,
    deterministic=False,
    train_datasets=['mpi3d-train', 'mpii-train'],
    lr_min=1e-1,
    lr_max=1e2,
    max_iters=1000,
    ema_beta=0.99,  # Beta for exponential moving average of loss
    weight_decay=0,
    momentum=0.9,
)


@ex.main
def sacred_main(_run: Run, seed, showoff, batch_size, model_desc, deterministic, train_datasets,
         lr_min, lr_max, max_iters, ema_beta, weight_decay, momentum):
    seed_all(seed)
    init_algorithms(deterministic=deterministic)

    model = create_model(model_desc).to(global_opts['device'])
    data_loader = create_train_dataloader(train_datasets, model.data_specs, batch_size,
                                          examples_per_epoch=(max_iters * batch_size))
    data_iter = iter(data_loader)

    print(json.dumps(model_desc, sort_keys=True, indent=2))

    def do_training_iteration(optimiser):
        batch = next(data_iter)

        in_var = batch['input'].to(global_opts['device'], torch.float32)
        target_var = batch['target'].to(global_opts['device'], torch.float32)
        mask_var = batch['joint_mask'].to(global_opts['device'], torch.float32)

        # Calculate predictions and loss
        out_var = model(in_var)
        loss = forward_loss(model, out_var, target_var, mask_var, batch['valid_depth'])

        # Calculate gradients
        optimiser.zero_grad()
        loss.backward()

        # Update parameters
        optimiser.step()

        return loss.item()

    optimiser = SGD(model.parameters(), lr=1, weight_decay=weight_decay, momentum=momentum)

    tel = tele.Telemetry({
        'config': ValueMeter(skip_reset=True),
        'host_info': ValueMeter(skip_reset=True),
        'loss_lr_fig': ValueMeter(),
    })

    notebook = None
    if showoff:
        title = 'Hyperparameter search ({}@{})'.format(model_desc['type'], model_desc['version'])
        notebook = create_showoff_notebook(title, ['lrfinder'])

        from tele.showoff import views

        tel.sink(tele.showoff.Conf(notebook), [
            views.Inspect(['config'], 'Experiment configuration', flatten=True),
            views.Inspect(['host_info'], 'Host information', flatten=True),
            views.FrameContent(['loss_lr_fig'], 'Loss vs learning rate graph', 'plotly'),
        ])

    def set_progress(value):
        if notebook is not None:
            notebook.set_progress(value)

    tel['config'].set_value(_run.config)
    tel['host_info'].set_value(get_host_info())

    lrs = np.geomspace(lr_min, lr_max, max_iters)
    losses = []
    avg_loss = 0
    min_loss = np.inf
    for i, lr in enumerate(tqdm(lrs, ascii=True)):
        set_progress(i / len(lrs))

        for param_group in optimiser.param_groups:
            param_group['lr'] = lr
        loss = do_training_iteration(optimiser)
        avg_loss = ema_beta * avg_loss + (1 - ema_beta) * loss
        smoothed_loss = avg_loss / (1 - ema_beta ** (i + 1))
        if min_loss > 0 and smoothed_loss > 4 * min_loss:
            break
        min_loss = min(smoothed_loss, min_loss)
        losses.append(smoothed_loss)

        if i % 10 == 0:
            fig = go.Figure(
                data=[go.Scatter(x=lrs[:len(losses)].tolist(), y=losses, mode='lines')],
                layout=go.Layout(
                    margin=go.Margin(l=60, r=40, b=80, t=20, pad=4),
                    xaxis=go.XAxis(title='Learning rate', type='log', exponentformat='power'),
                    yaxis=go.YAxis(title='Training loss'),
                )
            )
            tel['loss_lr_fig'].set_value(fig)
            tel.step()

    set_progress(1)


def main(argv, common_opts):
    global_opts.update(common_opts)
    ex.run_commandline(argv)


Hyperparams_Subcommand = Subcommand(name='hyperparams', func=main,
                                    help='search for good training hyperparameters')

if __name__ == '__main__':
    Hyperparams_Subcommand.run()
