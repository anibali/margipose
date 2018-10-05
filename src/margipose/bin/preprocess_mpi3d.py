#!/usr/bin/env python3

"""Preprocess data from the MPI-INF-3DHP dataset.

The input files may be obtained from http://gvv.mpi-inf.mpg.de/3dhp-dataset/.
"""

import argparse
import sys
from os import path, listdir

from margipose.data.mpi_inf_3dhp.preprocess import preprocess_training_data, \
    preprocess_validation_data, preprocess_training_masks, preprocess_validation_masks, \
    preprocess_test_data


def parse_args(argv):
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description='Preprocess data from the MPI-INF-3DHP dataset')
    parser.add_argument('-i', '--input-dir', type=str, metavar='DIR',
                        help='path to directory containing S1, S2, ...')
    parser.add_argument('-t', '--input-test-dir', type=str, metavar='DIR',
                        help='path to directory containing TS1, TS2, ...')
    parser.add_argument('-o', '--out-dir', type=str, metavar='DIR', required=True,
                        default='/data/fast/mpi_inf_3dhp',
                        help='directory to write preprocessed data to')

    args = parser.parse_args(argv[1:])

    return args


def assert_listing_contains(dir, expected):
    listing = listdir(dir)
    for child in expected:
        assert child in listing, '{} does not exist'.format(path.join(dir, child))


def main(argv=sys.argv):
    args = parse_args(argv)

    if args.input_dir:
        assert_listing_contains(args.input_dir, ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8'])

    if args.input_test_dir:
        assert_listing_contains(args.input_test_dir, ['TS1', 'TS2', 'TS3', 'TS4', 'TS5', 'TS6'])

    train_out_dir = path.join(args.out_dir, 'train')
    val_out_dir = path.join(args.out_dir, 'val')
    test_out_dir = path.join(args.out_dir, 'test')

    if args.input_dir:
        print('Starting data preprocessing.')
        print('This may take several hours to complete.')
        print()
        print('Training set')
        preprocess_training_data(args.input_dir, train_out_dir)
        print('Done')
        print()
        print('Validation set')
        preprocess_validation_data(args.input_dir, val_out_dir)
        print('Done')
        print()

    if args.input_test_dir:
        print('Test set')
        preprocess_test_data(args.input_test_dir, test_out_dir)
        print('Done')
        print()

    if args.input_dir:
        print('Starting mask preprocessing.')
        print()
        print('Training set')
        preprocess_training_masks(train_out_dir)
        print('Done')
        print()
        print('Validation set')
        preprocess_validation_masks(val_out_dir)
        print('Done')

    print()
    print('All preprocessing has completed.')


if __name__ == '__main__':
    main()
