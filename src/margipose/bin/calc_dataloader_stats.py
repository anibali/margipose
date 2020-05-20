import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys
import threading
from matplotlib import animation
from tqdm import tqdm

from margipose.data.skeleton import CanonicalSkeletonDesc
from margipose.models import create_model
from margipose.models.margipose_model import Default_MargiPose_Desc
from margipose.train_helpers import create_train_dataloader


def parse_args(parser: argparse.ArgumentParser, argv):
    if argv is None:
        argv = sys.argv[1:]
    return parser.parse_args(argv)


def init_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32,
                        help='number of examples per batch')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of passes through the training dataset')
    parser.add_argument('--dataset', type=str, default='mpi3d-train',
                        help='dataset to draw examples from')
    parser.add_argument('--examples-per-epoch', type=int, default=32000,
                        help='number of examples to draw per epoch')
    parser.add_argument('--with-image', action='store_true', default=False,
                        help='load images')
    parser.add_argument('--output', type=str,
                        help='output file to save figure to')
    return parser


class StatTracker:
    def __init__(self, bins):
        self.bins = bins
        self.bincounts = np.zeros(len(bins) + 1)
        self.min = np.inf
        self.max = -np.inf

    def add_samples(self, samples: np.ndarray):
        assert samples.ndim == 1
        self.min = min(self.min, samples.min())
        self.max = max(self.max, samples.max())
        indices = np.digitize(samples, self.bins)
        counts = np.bincount(indices)
        self.bincounts[:len(counts)] += counts

    def plot_hist(self, ax, **kwargs):
        n = self.bincounts.sum()
        if n > 0:
            freqs = self.bincounts / n
        else:
            freqs = self.bincounts
        return ax.plot(self.bins, freqs[:-1], **kwargs)

    def __repr__(self):
        fields = [
            f'min={self.min:0.4f}',
            f'max={self.max:0.4f}',
        ]
        summary = ', '.join(fields)
        return f'StatTracker[{summary}]'


def calculate_stats(stats, opts):
    model_desc = Default_MargiPose_Desc
    model = create_model(model_desc)
    skeleton = CanonicalSkeletonDesc
    loader = create_train_dataloader(
        [opts.dataset], model.data_specs, opts.batch_size, opts.examples_per_epoch, False)
    loader.dataset.without_image = not opts.with_image
    for epoch in range(opts.epochs):
        for batch in tqdm(loader, total=len(loader), leave=False, ascii=True):
            joints_3d = np.asarray(batch['target'])
            stats['root_x'].add_samples(joints_3d[:, skeleton.root_joint_id, 0])
            stats['root_y'].add_samples(joints_3d[:, skeleton.root_joint_id, 1])
            stats['root_z'].add_samples(joints_3d[:, skeleton.root_joint_id, 2])
            stats['lankle_x'].add_samples(joints_3d[:, skeleton.joint_names.index('left_ankle'), 0])
            stats['lankle_y'].add_samples(joints_3d[:, skeleton.joint_names.index('left_ankle'), 1])
            stats['lankle_z'].add_samples(joints_3d[:, skeleton.joint_names.index('left_ankle'), 2])
            if opts.with_image:
                image = np.asarray(batch['input'])
                stats['red'].add_samples(image[:, 0].ravel())
                stats['green'].add_samples(image[:, 1].ravel())
                stats['blue'].add_samples(image[:, 2].ravel())
            stats['index'].add_samples(np.asarray(batch['index'], dtype=np.float32) / (len(loader.dataset) - 1))
        tqdm.write(f'Epoch {epoch + 1:3d}')
        tqdm.write(repr(stats))
    tqdm.write('Done.')


def main(argv=None):
    opts = parse_args(init_argument_parser(), argv)

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2)
    ax3 = fig.add_subplot(4, 1, 3)
    ax4 = fig.add_subplot(4, 1, 4)

    # TODO: Add histogram for example index
    stats = {
        'root_x': StatTracker(np.linspace(-1.0, 1.0, 100)),
        'root_y': StatTracker(np.linspace(-1.0, 1.0, 100)),
        'root_z': StatTracker(np.linspace(-1.0, 1.0, 100)),
        'lankle_x': StatTracker(np.linspace(-1.0, 1.0, 100)),
        'lankle_y': StatTracker(np.linspace(-1.0, 1.0, 100)),
        'lankle_z': StatTracker(np.linspace(-1.0, 1.0, 100)),
        'red': StatTracker(np.linspace(-3.0, 1.0, 100)),
        'green': StatTracker(np.linspace(-3.0, 1.0, 100)),
        'blue': StatTracker(np.linspace(-3.0, 1.0, 100)),
        'index': StatTracker(np.linspace(0.0, 1.0, 100)),
    }

    thread = threading.Thread(target=calculate_stats, args=(stats,opts), daemon=True)
    thread.start()

    def draw_plots(_):
        ax1.clear()
        stats['root_x'].plot_hist(ax1, label='root_x')
        stats['root_y'].plot_hist(ax1, label='root_y')
        # stats['root_z'].plot_hist(ax1, label='root_z')
        ax1.legend()
        ax2.clear()
        stats['lankle_x'].plot_hist(ax2, label='lankle_x')
        stats['lankle_y'].plot_hist(ax2, label='lankle_y')
        stats['lankle_z'].plot_hist(ax2, label='lankle_z')
        ax2.legend()
        ax3.clear()
        stats['red'].plot_hist(ax3, label='red', c='red')
        stats['green'].plot_hist(ax3, label='green', c='green')
        stats['blue'].plot_hist(ax3, label='blue', c='blue')
        ax3.legend()
        ax4.clear()
        stats['index'].plot_hist(ax4, label='index')
        ax4.legend()

    if opts.output:
        thread.join()
        draw_plots(None)
        plt.savefig(opts.output)
    else:
        anim = animation.FuncAnimation(fig, draw_plots, interval=200)
        plt.show()


if __name__ == '__main__':
    main()
