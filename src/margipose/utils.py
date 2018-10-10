from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go

from PIL import ImageDraw
import torch
import numpy as np
import random
from time import perf_counter
from contextlib import contextmanager


def seed_all(seed):
    """Seed all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_algorithms(deterministic=False):
    if deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def _make_joint_metadata_fn(skel_desc):
    def joint_metadata_fn(joint_id):
        group = 'centre'
        if skel_desc.joint_names[joint_id].startswith('left_'):
            group = 'left'
        if skel_desc.joint_names[joint_id].startswith('right_'):
            group = 'right'
        return {
            'parent': skel_desc.joint_tree[joint_id],
            'group': group
        }
    return joint_metadata_fn


def plotly_skeleton_figure(skel3d, skel_desc):
    meta_fn = _make_joint_metadata_fn(skel_desc)

    cxs = []
    cys = []
    czs = []

    lxs = []
    lys = []
    lzs = []

    rxs = []
    rys = []
    rzs = []

    xt = list(skel3d[:, 0])
    zt = list(-skel3d[:, 1])
    yt = list(skel3d[:, 2])

    for j, p in enumerate(skel_desc.joint_tree):
        metadata = meta_fn(j)
        if metadata['group'] == 'left':
            xs, ys, zs = lxs, lys, lzs
        elif metadata['group'] == 'right':
            xs, ys, zs = rxs, rys, rzs
        else:
            xs, ys, zs = cxs, cys, czs

        xs += [xt[j], xt[p], None]
        ys += [yt[j], yt[p], None]
        zs += [zt[j], zt[p], None]

    points = go.Scatter3d(
        x=list(skel3d[:, 0]),
        z=list(-skel3d[:, 1]),
        y=list(skel3d[:, 2]),
        text=skel_desc.joint_names,
        mode='markers',
        marker=dict(color='grey', size=3, opacity=0.8),
    )

    centre_lines = go.Scatter3d(
        x=cxs,
        y=cys,
        z=czs,
        mode='lines',
        line=dict(color='magenta', width=1),
        hoverinfo='none',
    )

    left_lines = go.Scatter3d(
        x=lxs,
        y=lys,
        z=lzs,
        mode='lines',
        line=dict(color='blue', width=1),
        hoverinfo='none',
    )

    right_lines = go.Scatter3d(
        x=rxs,
        y=rys,
        z=rzs,
        mode='lines',
        line=dict(color='red', width=1),
        hoverinfo='none',
    )

    layout = go.Layout(
        margin=go.Margin(l=20, r=20, b=20, t=20, pad=0),
        hovermode='closest',
        scene=go.Scene(
            aspectmode='data',
            yaxis=go.YAxis(title='z'),
            zaxis=go.ZAxis(title='y'),
        ),
        showlegend=False,
    )
    fig = go.Figure(data=[points, centre_lines, left_lines, right_lines], layout=layout)

    return fig


def plot_skeleton_on_axes3d(skel, skel_desc, ax: Axes3D, invert=True, alpha=1.0):
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')

    # NOTE: y and z axes are swapped
    xs = skel.narrow(-1, 0, 1).numpy()
    ys = skel.narrow(-1, 2, 1).numpy()
    zs = skel.narrow(-1, 1, 1).numpy()

    # Correct aspect ratio (https://stackoverflow.com/a/21765085)
    max_range = np.array([
        xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()
    ]).max() / 2.0
    mid_x = (xs.max() + xs.min()) * 0.5
    mid_y = (ys.max() + ys.min()) * 0.5
    mid_z = (zs.max() + zs.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_aspect('equal')

    if invert:
        ax.invert_zaxis()

    # Set starting view
    ax.view_init(elev=20, azim=-100)

    get_joint_metadata = _make_joint_metadata_fn(skel_desc)
    for joint_id, joint in enumerate(skel):
        meta = get_joint_metadata(joint_id)
        color = 'magenta'
        if meta['group'] == 'left':
            color = 'blue'
        if meta['group'] == 'right':
            color = 'red'
        parent = skel[meta['parent']]
        offset = parent - joint
        ax.quiver(
            [joint[0]], [joint[2]], [joint[1]],
            [offset[0]], [offset[2]], [offset[1]],
            color=color,
            alpha=alpha,
        )

    ax.scatter(xs, ys, zs, color='grey', alpha=alpha)


def plot_skeleton_on_axes(skel, skel_desc, ax, alpha=1.0):
    get_joint_metadata = _make_joint_metadata_fn(skel_desc)
    for joint_id, joint in enumerate(skel):
        meta = get_joint_metadata(joint_id)
        color = 'magenta'
        if meta['group'] == 'left':
            color = 'blue'
        if meta['group'] == 'right':
            color = 'red'
        parent = skel[meta['parent']]
        offset = parent - joint
        if offset.norm(2) >= 1:
            ax.arrow(
                joint[0], joint[1],
                offset[0], offset[1],
                color=color,
                alpha=alpha,
                head_width=2,
                length_includes_head=True,
            )

    xs = skel.narrow(-1, 0, 1).numpy()
    ys = skel.narrow(-1, 1, 1).numpy()
    ax.scatter(xs, ys, color='grey', alpha=alpha)


def draw_skeleton_2d(img, skel2d, skel_desc, mask=None, width=1):
    assert skel2d.size(-1) == 2, 'coordinates must be 2D'
    draw = ImageDraw.Draw(img)
    get_joint_metadata = _make_joint_metadata_fn(skel_desc)
    for joint_id in range(skel_desc.n_joints):
        meta = get_joint_metadata(joint_id)
        color = (255, 0, 255)
        if meta['group'] == 'left':
            color = (0, 0, 255)
        if meta['group'] == 'right':
            color = (255, 0, 0)
        if mask is not None:
            if mask[joint_id] == 0 or mask[meta['parent']] == 0:
                color = (128, 128, 128)
        draw.line(
            [*skel2d[joint_id], *skel2d[meta['parent']]],
            color, width=width
        )


@contextmanager
def timer(meter, n=1):
    start_time = perf_counter()
    yield
    time_elapsed = perf_counter() - start_time
    meter.add(time_elapsed / n)


def generator_timer(iterable, meter):
    iterator = iter(iterable)
    while True:
        with timer(meter):
            vals = next(iterator)
        yield vals
