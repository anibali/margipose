#!/usr/bin/env python3

"""The main entrypoint for the `margipose` command."""

import sys

from margipose.cli import create_common_arg_parser
from .eval_3d import Eval_Subcommand
from .hyperparam_search import Hyperparams_Subcommand
from .infer_single import Infer_Subcommand
from .run_gui import GUI_Subcommand
from .train_3d import Train_Subcommand

_Subcommand_List = [
    GUI_Subcommand,
    Eval_Subcommand,
    Train_Subcommand,
    Hyperparams_Subcommand,
    Infer_Subcommand,
]
Subcommands = {subcmd.name: subcmd for subcmd in _Subcommand_List}


def main(argv=sys.argv):
    parser = create_common_arg_parser()
    subparsers = parser.add_subparsers(dest='subparser_name', title='subcommands')

    for subcmd in Subcommands.values():
        subparsers.add_parser(subcmd.name, add_help=False, help=subcmd.help)

    args, subargs = parser.parse_known_args(argv[1:])
    if args.subparser_name is not None:
        Subcommands[args.subparser_name].run([argv[0]] + subargs, args)
    else:
        parser.print_usage()


if __name__ == '__main__':
    main()
