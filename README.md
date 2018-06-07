# MargiPose

Accompanying PyTorch code for the paper
["3D Human Pose Estimation with 2D Marginal Heatmaps"](https://arxiv.org/abs/1806.01484).

## Setup

Requirements:

* Linux
* Docker
* Docker Compose
* `nvidia-docker2` (for GPU support)

### Prepare datasets

You only need to prepare the datasets that you are interested in using.

NOTE: Currently `docker-compose.yml` contains example volume mounts for the datasets.
You will need to edit the entries for datasets that you have prepared, and remove
the others.

#### Human3.6M

1. Use the scripts available at https://github.com/anibali/h36m-fetch to download
   and preprocess Human3.6M data.
2. Edit the volume mounts in `docker-compose.yml` so that the absolute location of
   the `processed/` directory created by h36m-fetch is bound to `/datasets/h36m`
   inside the Docker container.

#### MPI-INF-3DHP

1. Download [the original MPI-INF-3DHP dataset](http://gvv.mpi-inf.mpg.de/3dhp-dataset/).
2. Use the `bin/preprocess_mpi3d.py` script to preprocess the data.
3. Edit the volume mounts in `docker-compose.yml` so that the absolute location of
   the processed MPI-INF-3DHP data is bound to `/datasets/mpi3d` inside the Docker container.

#### MPII

1. Create a directory to hold the MPII data.
2. From inside your MPII data directory, download the annotation files:
    ```bash
    curl -L https://raw.githubusercontent.com/umich-vl/pose-hg-train/master/data/mpii/annot.h5 > mpii_annot_all.h5
    curl -L https://raw.githubusercontent.com/umich-vl/pose-hg-train/master/data/mpii/annot/valid.h5 > mpii_annot_valid.h5
    ```
3. Download [the original MPII dataset](http://human-pose.mpi-inf.mpg.de/) and move the
   `images/` folder into your MPII data directory.
4. Edit the volume mounts in `docker-compose.yml` so that the absolute location of
the MPII data is bound to `/datasets/mpii` inside the Docker container.

### [Optional] Configure and run Showoff

Showoff is a display server which allows you to visualise model training progression.
The following steps guide you through starting a Showoff server and configuring
MargiPose to use it.

1. Change `POSTGRES_PASSWORD` in `showoff/postgres.env`. Using a randomly generated password is
   recommended.
2. Change `COOKIE_SECRET` in `showoff/showoff.env`. Once again, using a randomly generated
   value is recommended.
3. From a terminal in the showoff directory, run `docker-compose up -d showoff`. This will
   start the Showoff server.
4. Log in using the username "admin" and the password "password".
5. Change the admin password.
6. Open up `showoff/showoff-client.env` in a text editor.
7. From the Showoff account page, add a new API key. Copy the API key ID and secret key
   into `showoff-client.env` (you will need to uncomment the appropriate lines).

## Running scripts

A `run.sh` launcher script is provided, which will run any command within a Docker container
containing all of MargiPose's dependencies. Here are a few examples.

Train a MargiPose model on the mpi3d dataset:

```bash
./run.sh bin/train_3d.py with margipose 1cycle mpi3d
```

Evaluate a model's test set performance using the second GPU:

```bash
NVIDIA_VISIBLE_DEVICES=1 ./run.sh bin/eval_3d.py --model margipose-mpi3d.pth --dataset mpi3d-test
```

Explore qualitative results with a GUI:

```bash
./run.sh bin/run_gui.py --model margipose-mpi3d.pth --dataset mpi3d-test
```

Run the tests:

```bash
./run.sh python setup.py test
```

## License and citation

(C) 2018 Aiden Nibali

This project is open source under the terms of the Apache License 2.0.

If you use any part of this work in a research project, please cite the following paper:

```
@article{nibali2018margipose,
  title={3D Human Pose Estimation with 2D Marginal Heatmaps},
  author={Nibali, Aiden and He, Zhen and Morgan, Stuart and Prendergast, Luke},
  journal={arXiv preprint arXiv:1806.01484},
  year={2018}
}
```
