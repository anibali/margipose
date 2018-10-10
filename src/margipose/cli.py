import argparse
import sys

import torch


def create_common_arg_parser():
    """Create an argument parser for the root CLI command shared by subcommands."""
    parser = argparse.ArgumentParser(prog='margipose',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device', help='primary computation device, eg "cpu" or "cuda:0"',
                        default='cuda:0')
    return parser


Default_Common_Args = create_common_arg_parser().parse_args([])


def common_args_to_opts(common_args):
    """Parse common_args into a dict of Python objects."""
    opts = dict(
        device=torch.device(common_args.device),
    )
    return opts


class Subcommand:
    def __init__(self, name, func, help=None):
        self.name = name
        self.func = func
        self.help = help

    def run(self, argv=None, common_args=None):
        if argv is None:
            argv = sys.argv
        if common_args is None:
            common_args = Default_Common_Args
        return self.func(argv, common_args_to_opts(common_args))
