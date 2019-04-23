#!/usr/bin/env python3

"""Perform 3D pose inference on a single image.

The image is assumed to be centred on a human subject and scaled appropriately.

Since the camera intrinsics are not known, the skeleton will be shown in normalized form.
This means that bones may be warped due to non-reversed transformations.
"""

import argparse

import PIL.Image
import matplotlib.pylab as plt
import torch
from mpl_toolkits.mplot3d import Axes3D
from pose3d_utils.coords import ensure_cartesian

from margipose.cli import Subcommand
from margipose.data.skeleton import CanonicalSkeletonDesc
from margipose.data_specs import ImageSpecs
from margipose.models import load_model
from margipose.utils import seed_all, init_algorithms, plot_skeleton_on_axes3d

CPU = torch.device('cpu')


def parse_args(argv):
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(prog='margipose-infer',
                                     description='3D human pose inference',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, metavar='FILE', default=argparse.SUPPRESS,
                        required=True,
                        help='path to model file')
    parser.add_argument('--image', type=str, metavar='FILE', default=argparse.SUPPRESS,
                        required=True,
                        help='image file to infer pose from')
    parser.add_argument('--multicrop', action='store_true',
                        help='enable the use of multiple crops')

    args = parser.parse_args(argv[1:])

    return args


def main(argv, common_opts):
    args = parse_args(argv)
    seed_all(12345)
    init_algorithms(deterministic=True)
    torch.set_grad_enabled(False)

    device = common_opts['device']

    assert args.multicrop == False, 'TODO: Implement multi-crop for single image inference.'

    model = load_model(args.model).to(device).eval()

    input_specs: ImageSpecs = model.data_specs.input_specs

    image: PIL.Image.Image = PIL.Image.open(args.image, 'r')
    image.thumbnail((input_specs.width, input_specs.height))
    inp = input_specs.convert(image).to(device, torch.float32)

    output = model(inp[None, ...])[0]

    norm_skel3d = ensure_cartesian(output.to(CPU, torch.float64), d=3)

    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2: Axes3D = fig.add_subplot(1, 2, 2, projection='3d')

    ax1.imshow(input_specs.unconvert(inp.to(CPU)))
    plot_skeleton_on_axes3d(norm_skel3d, CanonicalSkeletonDesc, ax2, invert=True)

    plt.show()


Infer_Subcommand = Subcommand(name='infer', func=main, help='infer 3D pose for single image')

if __name__ == '__main__':
    Infer_Subcommand.run()

