from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go

from PIL import ImageDraw
import torch
from torch.autograd import Variable
import numpy as np
import random
from time import perf_counter
from contextlib import contextmanager
from subprocess import check_output


def seed_all(seed):
    """Seed all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def draw_skeleton(img, skel, skel_desc, intrinsics, mask=None):
    """Project the pose into 2D and draw it over the image."""
    skel2d = intrinsics.project_cartesian(skel)
    draw_skeleton_2d(img, skel2d, skel_desc, mask)


def draw_canonical_skeleton(img, joints3d, intrinsics):
    from margipose.data.skeleton import CanonicalSkeletonDesc
    draw_skeleton(img, joints3d, CanonicalSkeletonDesc, intrinsics)


def draw_wireframe_2d(img, vertices, edges, color=(255, 255, 255)):
    assert vertices.size(-1) == 2, 'coordinates must be 2D'
    draw = ImageDraw.Draw(img)
    for i, j in edges:
        draw.line([*list(vertices[i]), *list(vertices[j])], color, width=1)


def draw_wireframe(img, intrinsics, vertices, edges, color=(255, 255, 255)):
    vertices_2d = intrinsics.project_cartesian(vertices)
    draw_wireframe_2d(img, vertices_2d, edges, color)


def unit_quad():
    vertices = [
        [-1, -1, 0, 1],  # 0
        [-1, 1, 0, 1],   # 1
        [1, -1, 0, 1],   # 2
        [1, 1, 0, 1],    # 3
    ]
    edges = [(0, 1), (1, 3), (3, 2), (2, 0)]
    return torch.DoubleTensor(vertices), edges


def draw_quad(img, intrinsics, centre, width, height):
    """Draws a camera-facing quad."""
    quad_verts, quad_edges = unit_quad()
    quad_verts[:, 0] *= width / 2
    quad_verts[:, 1] *= height / 2
    quad_verts[:, :3] += centre
    draw_wireframe(img, intrinsics, quad_verts, quad_edges, color=(255, 255, 255))


def unit_cube():
    vertices = [
        [-1, -1, -1, 1],  # 0
        [-1, -1, 1, 1],   # 1
        [-1, 1, -1, 1],   # 2
        [-1, 1, 1, 1],    # 3
        [1, -1, -1, 1],   # 4
        [1, -1, 1, 1],    # 5
        [1, 1, -1, 1],    # 6
        [1, 1, 1, 1],     # 7
    ]
    edges = [
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
    ]
    return torch.DoubleTensor(vertices), edges


def draw_cube(img, intrinsics, centre, width, height, depth):
    """Draws a camera-facing cube."""
    cube_verts, cube_edges = unit_cube()
    cube_verts[:, 0] *= width / 2
    cube_verts[:, 1] *= height / 2
    cube_verts[:, 2] *= depth / 2
    cube_verts[:, :3] += centre
    draw_wireframe(img, intrinsics, cube_verts, cube_edges, color=(128, 128, 128))


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


def object_memory_usage(obj):
    if isinstance(obj, Variable):
        obj = obj.data
    if torch.is_tensor(obj):
        obj = obj.storage()
    if torch.is_storage(obj):
        return obj.size() * obj.element_size()
    raise Exception('unrecognised object type')


def gpu_memory_usage():
    torch.cuda.empty_cache()
    result = check_output([
        'nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'
    ], encoding='utf-8')
    return sum(map(int, result.strip().split('\n'))) * 1024**2
