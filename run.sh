#!/usr/bin/env bash
#
# Use this launcher script to run executables in a containerised environment.

set -e

echo "Building Docker images"
docker-compose build > /dev/null
echo "Successfully built Docker images"
echo

NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"
export NVIDIA_VISIBLE_DEVICES
docker-compose run --user="$(id -u):$(id -g)" -e NVIDIA_VISIBLE_DEVICES --rm main "$@"
