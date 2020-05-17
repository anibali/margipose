#!/usr/bin/env python3

"""Export a trained model for sharing."""


import argparse
import sys

import torch
from torch import onnx

from margipose.models import create_model
from margipose.utils import seed_all, init_algorithms


def parse_args(argv):
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description='3D pose estimation model exporter')
    parser.add_argument('-i', '--input', type=str, metavar='FILE', required=True,
                        help='path to input model file')
    parser.add_argument('-o', '--output', type=str, metavar='FILE', required=True,
                        help='desired path to output model file')
    parser.add_argument('-f', '--format', type=str, default='pytorch', choices=['pytorch', 'onnx'],
                        help='format of output model')

    args = parser.parse_args(argv[1:])

    return args


def main(argv=sys.argv):
    args = parse_args(argv)
    seed_all(12345)
    init_algorithms(deterministic=True)

    # Load the model into system memory (CPU, not GPU)
    model_state = torch.load(args.input, map_location='cpu')
    model_desc = model_state['model_desc']
    model = create_model(model_desc)
    model.load_state_dict(model_state['state_dict'])
    model.eval()

    if args.format == 'pytorch':
        new_model_state = {
            'state_dict': model.state_dict(),
            'model_desc': model_desc,
            'train_datasets': model_state.get('train_datasets', []),
        }
        torch.save(new_model_state, args.output)
    elif args.format == 'onnx':
        image_height = model.data_specs.input_specs.height
        image_width = model.data_specs.input_specs.width
        dummy_input = torch.randn(1, 3, image_height, image_width)
        onnx.export(model, (dummy_input,), args.output, verbose=False)
    else:
        raise Exception('Unrecognised model format: {}'.format(args.format))


if __name__ == '__main__':
    main()
