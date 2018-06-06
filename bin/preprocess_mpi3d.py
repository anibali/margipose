#!/usr/bin/env python3

from margipose.data.mpi_inf_3dhp.preprocess import preprocess_training_data, \
    preprocess_validation_data, preprocess_training_masks, preprocess_validation_masks, \
    preprocess_test_data


def main():
    in_dir = '/data/mpi-inf-3dhp'
    test_in_dir = '/data/mpi-inf-3dhp/mpi_inf_3dhp_test_set'
    train_out_dir = '/data/fast/mpi_inf_3dhp/train'
    val_out_dir = '/data/fast/mpi_inf_3dhp/val'
    test_out_dir = '/data/fast/mpi_inf_3dhp/test'

    print('Starting data preprocessing.')
    print('This may take several hours to complete.')
    print()
    print('Training set')
    preprocess_training_data(in_dir, train_out_dir)
    print('Done')
    print()
    print('Validation set')
    preprocess_validation_data(in_dir, val_out_dir)
    print('Done')
    print()
    print('Test set')
    preprocess_test_data(test_in_dir, test_out_dir)
    print('Done')

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
