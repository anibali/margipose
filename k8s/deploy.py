#!/usr/bin/env python3

"""Script for deploying MargiPose jobs to a Kubernetes cluster.

Examples:

```bash
# Train MargiPose on MPI-INF-3DHP data for 150 epochs
$ k8s/deploy.py --name=train-margipose-mpi3d -- \
  margipose train with margipose_model 1cycle mpi3d epochs=150

# Train MargiPose on Human3.6M data for 150 epochs
$ k8s/deploy.py --name=train-margipose-h36m -- \
  margipose train with margipose_model 1cycle h36m epochs=150
```
"""

import argparse
import json
import logging
import sys
from distutils.util import strtobool
from time import sleep

import docker
from kubernetes import client, config
from kubernetes.stream import stream

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('margipose-deploy')


def build_and_push_image(image_tag):
    docker_client = docker.from_env()
    log.info('Building Docker image...')
    docker_client.images.build(path='.', tag=image_tag)
    log.info('Pushing Docker image...')
    docker_client.images.push(image_tag)


def _host_volume(name, path, type):
    return client.V1Volume(
        name=name,
        host_path=client.V1HostPathVolumeSource(path=path, type=type)
    )


def deploy(args):
    build_and_push_image(args.image_tag)

    config.load_kube_config()

    namespace = config.list_kube_config_contexts()[1]['context']['namespace']

    meta = client.V1ObjectMeta(
        name=args.name,
        labels={'app.kubernetes.io/managed-by': 'margipose-deploy'},
    )

    container = client.V1Container(
        name='margipose',
        image=args.image_tag,
        image_pull_policy='Always',
        resources=client.V1ResourceRequirements(
            limits={'nvidia.com/gpu': '1'},
        ),
        args=args.command,
        tty=True,
        volume_mounts=[
            client.V1VolumeMount(mount_path='/datasets/h36m', name='h36m', read_only=True),
            client.V1VolumeMount(mount_path='/datasets/mpi3d', name='mpi3d', read_only=True),
            client.V1VolumeMount(mount_path='/datasets/raw-mpi3d', name='raw-mpi3d', read_only=True),
            client.V1VolumeMount(mount_path='/datasets/mpii', name='mpii', read_only=True),
            client.V1VolumeMount(mount_path='/home/user/.cache/torch/checkpoints', name='pretrained-models', read_only=False),
            client.V1VolumeMount(mount_path='/app/out', name='output', read_only=False),
        ],
    )

    pod_spec = client.V1PodSpec(
        restart_policy='Never',
        host_ipc=True,
        containers=[container],
        volumes=[
            _host_volume(
                name='h36m',
                path='/nfs/datasets/public/_Old/Human3.6M/processed',
                type='Directory',
            ),
            _host_volume(
                name='mpi3d',
                path='/nfs/datasets/public/_Old/MPI-INF-3DHP',
                type='Directory',
            ),
            _host_volume(
                name='raw-mpi3d',
                path='/nfs/datasets/public/MPI-INF-3DHP',
                type='Directory',
            ),
            _host_volume(
                name='mpii',
                path='/nfs/datasets/public/MPII_Human_Pose',
                type='Directory',
            ),
            _host_volume(
                name='pretrained-models',
                path='/nfs/models/pytorch',
                type='DirectoryOrCreate',
            ),
            _host_volume(
                name='output',
                path='/nfs/users/aiden/margipose-out',
                type='DirectoryOrCreate',
            ),
        ]
    )

    if args.node:
        pod_spec.node_selector = {
            'kubernetes.io/hostname': args.node,
        }

    pod = client.V1Pod(
        api_version='v1',
        kind='Pod',
        metadata=meta,
        spec=pod_spec,
    )

    v1 = client.CoreV1Api()

    log.debug('pod resource request: {}'.format(
        json.dumps(v1.api_client.sanitize_for_serialization(pod))
    ))

    v1.create_namespaced_pod(namespace, body=pod)

    log.info('A new pod has been created.\n' +
             '* Attach: kubectl attach -i {}\n'.format(args.name) +
             '* Delete: kubectl delete pod {}'.format(args.name))

    # log.info(v1.list_namespaced_pod(
    #     namespace,
    #     label_selector='app.kubernetes.io/managed-by=margipose-deploy'
    # ))

    log.info('Waiting for container to start...')
    phase = 'Pending'
    while phase in {'Pending', 'Unknown'}:
        phase = v1.read_namespaced_pod_status(args.name, namespace).status.phase

    if phase == 'Running':
        log.info('Attaching to running container...')
        resp = stream(
            v1.connect_get_namespaced_pod_attach,
            args.name,
            namespace,
            stdin=False,
            stdout=True,
            stderr=True,
            tty=True,
            _preload_content=False,
        )
        while resp.is_open():
            resp.update(timeout=1)
            if resp.peek_stdout():
                sys.stdout.write(resp.read_stdout())
                sys.stdout.flush()
            if resp.peek_stderr():
                sys.stderr.write(resp.read_stderr())
                sys.stderr.flush()
        resp.close()

    sleep(1)
    phase = v1.read_namespaced_pod_status(args.name, namespace).status.phase
    if phase in {'Succeeded', 'Failed'} and args.rm:
        log.info('Deleting pod...')
        v1.delete_namespaced_pod(args.name, namespace, client.V1DeleteOptions(
            api_version='v1', kind='DeleteOptions'))

    log.info('Done.')


def parse_args():
    """Parse command-line arguments."""

    def _bool(val):
        return bool(strtobool(val))

    parser = argparse.ArgumentParser(description='Deploy MargiPose jobs to Kubernetes',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, required=True,
                        help='name of the pod')
    parser.add_argument('--image-tag', type=str,
                        default='registry.dl.cs.latrobe.edu.au/aiden/margipose',
                        help='tag for the Docker image')
    parser.add_argument('--rm', type=_bool, const=True, default=False, nargs='?',
                        help='remove the pod after termination')
    parser.add_argument('--node', type=str,
                        help='specific cluster node to target for deployment')
    parser.add_argument('command', nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if len(args.command) > 0 and args.command[0] == '--':
        args.command = args.command[1:]

    return args


def main():
    args = parse_args()

    deploy(args)


if __name__ == '__main__':
    main()
