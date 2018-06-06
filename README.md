## Configuration

1. Copy the `.env.example` files in `secrets/` to `.env` files and fill in the values.
2. Edit `docker-compose.yml` so that the volume mounts for data point to their locations
   on your host filesystem. Currently they all point to locations under `/data/`.

## Datasets

### MPI-INF-3DHP

Original everything, requires preprocessing with `bin/preprocess_mpi3d.py`.

Note that originally test set subjects TS3 and TS4 had a slight temporal misalignment (2-3 frames)
between annotations and images. This was corrected in the dataset on 2018-03-02.

### MPII

Original images, anewell's annotations.

### H3.6M

Preprocessed with https://github.com/anibali/h36m-fetch.

## Running scripts

Use the `run.sh` launcher script, which will run the command within a Docker container. Here's
an example:

```bash
./run.sh bin/train_3d.py with margipose
```

## Running the tests

```bash
./run.sh python setup.py test
```
