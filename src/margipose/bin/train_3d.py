#!/usr/bin/env python3

import datetime
import json
from os import environ, path, makedirs
from collections import namedtuple

import sacred
import tele
import torch
from torch import optim
from pose3d_utils.coords import ensure_homogeneous
from sacred.host_info import get_host_info
from sacred.run import Run
from tele.meter import ValueMeter, MeanValueMeter

from margipose.cli import Subcommand
from margipose.data.mpi_inf_3dhp import MpiInf3dDataset
from margipose.dsntnn import average_loss
from margipose.hyperparam_scheduler import make_1cycle
from margipose.models import create_model
from margipose.models.margipose_model import Default_MargiPose_Desc
from margipose.models.chatterbox_model import Default_Chatterbox_Desc
from margipose.train_helpers import visualise_predictions, progress_iter, create_showoff_notebook, \
    learning_schedule, create_train_dataloader, create_val_dataloader
from margipose.utils import seed_all, init_algorithms, timer, generator_timer

sacred.SETTINGS['DISCOVER_SOURCES'] = 'dir'
ex = sacred.Experiment(base_dir=path.realpath(path.join(__file__, '..', '..')))

CPU = torch.device('cpu')

global_opts = {}


class Reporter:
    """Helper class for reporting training metrics."""

    def __init__(self, with_val=True):
        meters = {
            'config': ValueMeter(skip_reset=True),
            'host_info': ValueMeter(skip_reset=True),
            'epoch': ValueMeter(),
            'data_load_time': MeanValueMeter(),
            'data_transfer_time': MeanValueMeter(),
            'forward_time': MeanValueMeter(),
            'backward_time': MeanValueMeter(),
            'optim_time': MeanValueMeter(),
            'eval_time': MeanValueMeter(),
            'train_loss': MeanValueMeter(),
            'train_mpjpe': MeanValueMeter(),
            'train_pck': MeanValueMeter(),
            'train_examples': ValueMeter(),
        }
        if with_val:
            meters.update({
                'val_loss': MeanValueMeter(),
                'val_mpjpe': MeanValueMeter(),
                'val_pck': MeanValueMeter(),
                'val_examples': ValueMeter(),
            })
        self.with_val = with_val
        self.telemetry = tele.Telemetry(meters)

    def setup_console_output(self):
        """Setup stdout reporting output."""

        from tele.console import views
        if self.with_val:
            meters_to_print = ['train_loss', 'val_loss', 'train_pck', 'val_pck', 'val_mpjpe']
        else:
            meters_to_print = ['train_loss', 'train_pck']
        self.telemetry.sink(tele.console.Conf(), [
            views.KeyValue([mn]) for mn in meters_to_print
        ])

    def setup_folder_output(self, out_dir):
        """Setup file system reporting output."""
        pass

    def setup_showoff_output(self, notebook):
        """Setup Showoff reporting output."""

        from tele.showoff import views

        if self.with_val:
            maybe_val_views = [
                views.Images(['val_examples'], 'Validation example images', images_per_row=2),
                views.PlotlyLineGraph(['train_loss', 'val_loss'], 'Loss'),
                views.PlotlyLineGraph(['train_mpjpe', 'val_mpjpe'], '3D MPJPE'),
                views.PlotlyLineGraph(['train_pck', 'val_pck'], '3D PCK@150mm'),
            ]
        else:
            maybe_val_views = [
                views.PlotlyLineGraph(['train_loss'], 'Loss'),
                views.PlotlyLineGraph(['train_mpjpe'], '3D MPJPE'),
                views.PlotlyLineGraph(['train_pck'], '3D PCK@150mm'),
            ]
        self.telemetry.sink(tele.showoff.Conf(notebook), [
            views.Inspect(['config'], 'Experiment configuration', flatten=True),
            views.Inspect(['host_info'], 'Host information', flatten=True),
            views.Images(['train_examples'], 'Training example images', images_per_row=2),
            *maybe_val_views,
            views.PlotlyLineGraph(
                ['data_load_time', 'data_transfer_time', 'forward_time',
                 'backward_time', 'optim_time', 'eval_time'],
                'Training time breakdown'
            )
        ])

    def setup_sacred_output(self, run: Run):
        from tele.sacred import views
        scalars = ['train_loss', 'train_mpjpe', 'train_pck']
        if self.with_val:
            scalars += ['val_loss', 'val_mpjpe', 'val_pck']
        self.telemetry.sink(tele.sacred.Conf(run), [views.Scalar(scalars)])


def calculate_performance_metrics(batch, dataset, norm_pred_skels, mpjpe_meter, pck_meter):
    metrics = dataset.evaluate_3d_batch(batch, norm_pred_skels)
    for m in metrics:
        mpjpe_meter.add(m['mpjpe'])
        pck_meter.add(m['pck'])


@ex.capture
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


def do_training_pass(epoch, model, tel, loader, scheduler, on_progress):
    if hasattr(scheduler, 'step'):
        scheduler.step(epoch)
    optimiser = scheduler.optimizer

    vis_images = None
    samples_processed = 0

    model.train()
    for batch in generator_timer(progress_iter(loader, 'Training'), tel['data_load_time']):
        if hasattr(scheduler, 'batch_step'):
            scheduler.batch_step()

        with timer(tel['data_transfer_time']):
            in_var = batch['input'].to(global_opts['device'], torch.float32)
            target_var = batch['target'].to(global_opts['device'], torch.float32)
            mask_var = batch['joint_mask'].to(global_opts['device'], torch.float32)

        # Calculate predictions and loss
        with timer(tel['forward_time']):
            out_var = model(in_var)
            loss = forward_loss(model, out_var, target_var, mask_var, batch['valid_depth'])
            tel['train_loss'].add(loss.sum().item())

        # Calculate accuracy metrics
        with timer(tel['eval_time']):
            calculate_performance_metrics(
                batch,
                loader.dataset,
                ensure_homogeneous(out_var.to(CPU, torch.float64).detach(), d=3),
                tel['train_mpjpe'],
                tel['train_pck']
            )

        # Calculate gradients
        with timer(tel['backward_time']):
            optimiser.zero_grad()
            loss.backward()

        # Update parameters
        with timer(tel['optim_time']):
            optimiser.step()

        # Update progress
        samples_processed += len(batch['input'])
        on_progress(samples_processed)

        if vis_images is None:
            preds = out_var.to(CPU, torch.float64).detach()
            vis_images = visualise_predictions(preds, batch, loader.dataset)

    tel['train_examples'].set_value(vis_images[:8])


def do_validation_pass(epoch, model, tel, loader):
    vis_images = None

    model.eval()
    with torch.no_grad():
        for batch in progress_iter(loader, 'Validation'):
            in_var = batch['input'].to(global_opts['device'], torch.float32)
            target_var = batch['target'].to(global_opts['device'], torch.float32)
            mask_var = batch['joint_mask'].to(global_opts['device'], torch.float32)

            # Calculate predictions and loss
            out_var = model(in_var)
            loss = forward_loss(model, out_var, target_var, mask_var, batch['valid_depth'])
            tel['val_loss'].add(loss.sum().item())

            calculate_performance_metrics(
                batch,
                loader.dataset,
                ensure_homogeneous(out_var.to(CPU, torch.float64).detach(), d=3),
                tel['val_mpjpe'],
                tel['val_pck']
            )

            if vis_images is None:
                preds = out_var.to(CPU, torch.float64).detach()
                vis_images = visualise_predictions(preds, batch, loader.dataset)

    tel['val_examples'].set_value(vis_images[:8])


# Predefined model configuration sets
ex.add_named_config('margipose_model', model_desc=Default_MargiPose_Desc)
ex.add_named_config('chatterbox_model', model_desc=Default_Chatterbox_Desc)

# Predefined optimiser configuration sets
ex.add_named_config('rmsprop', optim_algorithm='rmsprop', epochs=150, lr=2.5e-3,
                    lr_milestones=[80, 140], lr_gamma=0.1)
ex.add_named_config('1cycle', optim_algorithm='1cycle', epochs=150, lr=1.0,
                    lr_milestones=None, lr_gamma=None)
ex.add_named_config('sgd_simple', optim_algorithm='sgd_simple', epochs=150, lr=0.2,
                    lr_milestones=None, lr_gamma=None)

# Predefined dataset configuration sets
ex.add_named_config('mpi3d', train_datasets=['mpi3d-trainval', 'mpii-trainval'], val_datasets=[])
ex.add_named_config('h36m', train_datasets=['h36m-trainval', 'mpii-trainval'], val_datasets=[])

# Configuration for a quick run (useful for debugging)
ex.add_named_config('quick', out_dir='', epochs=10, tags=['quick'], quick=True,
                    train_examples=256, val_examples=128)

# Configuration defaults
ex.add_config(
    **ex.named_configs['1cycle'](),
    showoff=not not environ.get('SHOWOFF_URL'),
    out_dir='out',
    batch_size=32,
    tags=[],
    quick=False,
    experiment_id=datetime.datetime.now().strftime('%Y%m%d-%H%M%S%f'),
    weights=None,
    deterministic=False,
    train_examples=32000,
    val_examples=1600,
    use_aug=True,
    preserve_root_joint_at_univ_scale=False,
)


@ex.main
def sacred_main(_run: Run, seed, showoff, out_dir, batch_size, epochs, tags, model_desc,
         experiment_id, weights, train_examples, val_examples, deterministic,
         train_datasets, val_datasets, lr, lr_milestones, lr_gamma, optim_algorithm,
         use_aug, preserve_root_joint_at_univ_scale):
    seed_all(seed)
    init_algorithms(deterministic=deterministic)

    exp_out_dir = None
    if out_dir:
        exp_out_dir = path.join(out_dir, experiment_id)
        makedirs(exp_out_dir, exist_ok=True)
    print(f'Experiment ID: {experiment_id}')

    ####
    # Model
    ####

    if weights is None:
        model = create_model(model_desc)
    else:
        details = torch.load(weights)
        model_desc = details['model_desc']
        model = create_model(model_desc)
        model.load_state_dict(details['state_dict'])
    model.to(global_opts['device'])

    print(json.dumps(model_desc, sort_keys=True, indent=2))

    ####
    # Data
    ####

    MpiInf3dDataset.preserve_root_joint_at_univ_scale = preserve_root_joint_at_univ_scale

    train_loader = create_train_dataloader(
        train_datasets, model.data_specs, batch_size, train_examples, use_aug)
    if len(val_datasets) > 0:
        val_loader = create_val_dataloader(
            val_datasets, model.data_specs, batch_size, val_examples)
    else:
        val_loader = None

    ####
    # Reporting
    ####

    reporter = Reporter(with_val=(val_loader is not None))

    reporter.setup_console_output()
    reporter.setup_sacred_output(_run)

    notebook = None
    if showoff:
        title = '3D pose model ({}@{})'.format(model_desc['type'], model_desc['version'])
        notebook = create_showoff_notebook(title, tags)
        reporter.setup_showoff_output(notebook)

    def set_progress(value):
        if notebook is not None:
            notebook.set_progress(value)

    tel = reporter.telemetry

    tel['config'].set_value(_run.config)
    tel['host_info'].set_value(get_host_info())

    ####
    # Optimiser
    ####

    if optim_algorithm == '1cycle':
        optimiser = optim.SGD(model.parameters(), lr=0)
        scheduler = make_1cycle(optimiser, epochs * len(train_loader), lr_max=lr, momentum=0.9)
    elif optim_algorithm == 'sgd_simple':
        optimiser = optim.SGD(model.parameters(), lr=lr, momentum=0)
        DummyScheduler = namedtuple('DummyScheduler', 'optimizer')
        scheduler = DummyScheduler(optimizer=optimiser)
    else:
        scheduler = learning_schedule(
            model.parameters(), optim_algorithm, lr, lr_milestones, lr_gamma)

    ####
    # Training
    ####

    model_file = None
    if exp_out_dir:
        model_file = path.join(exp_out_dir, 'model-latest.pth')
        with open(path.join(exp_out_dir, 'config.json'), 'w') as f:
            json.dump(tel['config'].value(), f, sort_keys=True, indent=2)

    for epoch in range(epochs):
        tel['epoch'].set_value(epoch)
        print('> Epoch {:3d}/{:3d}'.format(epoch + 1, epochs))

        def on_train_progress(samples_processed):
            so_far = epoch * len(train_loader.dataset) + samples_processed
            total = epochs * len(train_loader.dataset)
            set_progress(so_far / total)

        do_training_pass(epoch, model, tel, train_loader, scheduler, on_train_progress)
        if val_loader:
            do_validation_pass(epoch, model, tel, val_loader)

        _run.result = tel['train_pck'].value()[0]

        if model_file is not None:
            state = {
                'state_dict': model.state_dict(),
                'model_desc': model_desc,
                'train_datasets': train_datasets,
                'optimizer': scheduler.optimizer.state_dict(),
                'epoch': epoch + 1,
            }
            torch.save(state, model_file)

        tel.step()

    # Add the final model as a Sacred artifact
    if model_file is not None and path.isfile(model_file):
        _run.add_artifact(model_file)

    set_progress(1.0)
    return _run.result


def main(argv, common_opts):
    global_opts.update(common_opts)
    ex.run_commandline(argv)


Train_Subcommand = Subcommand(name='train', func=main, help='train a model')

if __name__ == '__main__':
    Train_Subcommand.run()
