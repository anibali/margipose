#!/usr/bin/env python3

import sys
import argparse

from margipose.bin import train_3d, eval_3d, run_gui


Subcommands = {
    'train': train_3d,
    'eval': eval_3d,
    'gui': run_gui,
}


def main(argv=sys.argv):
    parser = argparse.ArgumentParser(prog='margipose')
    subparsers = parser.add_subparsers(dest='subparser_name')

    for subcmd_name, subcmd_module in Subcommands.items():
        subparsers.add_parser(subcmd_name, add_help=False)

    args, subargs = parser.parse_known_args(argv[1:])
    if 'subparser_name' in args:
        Subcommands[args.subparser_name].main([argv[0]] + subargs)
    else:
        parser.print_usage()


if __name__ == '__main__':
    main()
