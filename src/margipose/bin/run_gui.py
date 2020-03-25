#!/usr/bin/env python3

import matplotlib

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import ttk
import tkinter.font
import argparse
import torch
from functools import lru_cache
import os
import numpy as np
from pose3d_utils.coords import ensure_homogeneous, ensure_cartesian

from margipose.data.get_dataset import get_dataset
from margipose.data.skeleton import absolute_to_root_relative, \
    VNect_Common_Skeleton, apply_rigid_alignment, CanonicalSkeletonDesc
from margipose.utils import plot_skeleton_on_axes3d, plot_skeleton_on_axes, seed_all, init_algorithms
from margipose.models import load_model
from margipose.eval import mpjpe, pck
from margipose.data_specs import DataSpecs, ImageSpecs, JointsSpecs
from margipose.cli import Subcommand


CPU = torch.device('cpu')


def parse_args(argv):
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(prog='margipose-gui',
                                     description='3D human pose browser GUI',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, metavar='FILE',
                        help='path to model file')
    parser.add_argument('--dataset', type=str, metavar='STR', default='mpi3d-test',
                        help='dataset name')

    args = parser.parse_args(argv[1:])

    return args


@lru_cache(maxsize=32)
def load_example(dataset, example_index):
    example = dataset[example_index]
    input = example['input']
    input_image = dataset.input_to_pil_image(input)
    camera = example['camera_intrinsic']
    transform_opts = example['transform_opts']
    gt_skel = None
    if 'target' in example:
        gt_skel = dict(original=example['original_skel'])
        gt_skel_norm = ensure_homogeneous(example['target'], d=3)
        gt_skel_denorm = dataset.denormalise_with_skeleton_height(gt_skel_norm, camera, transform_opts)
        gt_skel['image_space'] = camera.project_cartesian(gt_skel_denorm)
        gt_skel['camera_space'] = dataset.untransform_skeleton(gt_skel_denorm, transform_opts)
    return dict(
        input=input,
        input_image=input_image,
        camera=camera,
        transform_opts=transform_opts,
        gt_skel=gt_skel,
    )


@lru_cache(maxsize=32)
def load_and_process_example(dataset, example_index, device, model):
    example = load_example(dataset, example_index)
    if model is None:
        return example
    in_var = example['input'].unsqueeze(0).to(device, torch.float32)
    out_var = model(in_var)
    pred_skel_norm = ensure_homogeneous(out_var.squeeze(0).to(CPU, torch.float64), d=3)
    pred_skel_denorm = dataset.denormalise_with_skeleton_height(
        pred_skel_norm, example['camera'], example['transform_opts'])
    pred_skel_image_space = example['camera'].project_cartesian(pred_skel_denorm)
    pred_skel_camera_space = dataset.untransform_skeleton(pred_skel_denorm, example['transform_opts'])
    return dict(
        pred_skel=dict(
            normalised=pred_skel_norm,
            camera_space=pred_skel_camera_space,
            image_space=pred_skel_image_space,
        ),
        xy_heatmaps=[hm.squeeze(0).to(CPU, torch.float32) for hm in model.xy_heatmaps],
        zy_heatmaps=[hm.squeeze(0).to(CPU, torch.float32) for hm in model.zy_heatmaps],
        xz_heatmaps=[hm.squeeze(0).to(CPU, torch.float32) for hm in model.xz_heatmaps],
        **example
    )


def root_relative(skel):
    return absolute_to_root_relative(
        ensure_cartesian(skel, d=3),
        CanonicalSkeletonDesc.root_joint_id
    )


class MainGUIApp(tk.Tk):
    def __init__(self, dataset, device, model):
        super().__init__()

        self.dataset = dataset
        self.device = device
        self.model = model

        self.wm_title('3D pose estimation')
        self.geometry('1280x800')

        matplotlib.rcParams['savefig.format'] = 'svg'
        matplotlib.rcParams['savefig.directory'] = os.curdir

        # Variables
        self.var_cur_example = tk.StringVar()
        self.var_pred_visible = tk.IntVar(value=0)
        self.var_gt_visible = tk.IntVar(value=1)
        self.var_mpjpe = tk.StringVar(value='??')
        self.var_pck = tk.StringVar(value='??')
        self.var_aligned = tk.IntVar(value=0)
        self.var_joint = tk.StringVar(value='pelvis')

        if self.model is not None:
            self.var_pred_visible.set(1)

        global_toolbar = self._make_global_toolbar(self)
        global_toolbar.pack(side=tk.TOP, fill=tk.X)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=4, pady=4)
        def on_change_tab(event):
            self.update_current_tab()
        self.notebook.bind('<<NotebookTabChanged>>', on_change_tab)

        self.tab_update_funcs = [
            self._make_overview_tab(self.notebook),
            self._make_heatmap_tab(self.notebook),
        ]

        self.current_example_index = 0

    @property
    def current_example_index(self):
        return int(self.var_cur_example.get())

    @current_example_index.setter
    def current_example_index(self, value):
        self.var_cur_example.set(str(value))
        self.on_change_example()

    @property
    def pred_visible(self):
        return self.var_pred_visible.get() != 0

    @property
    def gt_visible(self):
        return self.var_gt_visible.get() != 0 and self.current_example['gt_skel'] is not None

    @property
    def is_aligned(self):
        return self.var_aligned.get() != 0

    def update_current_tab(self):
        cur_tab_index = self.notebook.index('current')

        if self.model is not None and self.current_example['gt_skel']:
            actual = root_relative(self.current_example['pred_skel']['camera_space'])
            expected = root_relative(self.current_example['gt_skel']['original'])

            if self.is_aligned:
                actual = apply_rigid_alignment(actual, expected)

            included_joints = [
                CanonicalSkeletonDesc.joint_names.index(joint_name)
                for joint_name in VNect_Common_Skeleton
            ]
            self.var_mpjpe.set('{:0.4f}'.format(mpjpe(actual, expected, included_joints)))
            self.var_pck.set('{:0.4f}'.format(pck(actual, expected, included_joints)))

        self.tab_update_funcs[cur_tab_index]()

    def _make_global_toolbar(self, master):
        toolbar = tk.Frame(master, bd=1, relief=tk.RAISED)

        def add_label(text):
            opts = dict(text=text) if isinstance(text, str) else dict(textvariable=text)
            label = tk.Label(toolbar, **opts)
            label.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)
            return label

        add_label('Example index:')
        txt_cur_example = tk.Spinbox(
            toolbar, textvariable=self.var_cur_example, command=self.on_change_example,
            wrap=True, from_=0, to=len(self.dataset) - 1, font=tk.font.Font(size=12))
        def on_key_cur_example(event):
            if event.keysym == 'Return':
                self.on_change_example()
        txt_cur_example.bind('<Key>', on_key_cur_example)
        txt_cur_example.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)

        if self.model is not None:
            add_label('MPJPE:')
            add_label(self.var_mpjpe)
            add_label('PCK@150mm:')
            add_label(self.var_pck)

            chk_aligned = tk.Checkbutton(
                toolbar, text='Procrustes alignment', variable=self.var_aligned,
                command=lambda: self.update_current_tab())
            chk_aligned.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)

        return toolbar

    def _make_overview_tab(self, notebook: ttk.Notebook):
        tab = tk.Frame(notebook)
        notebook.add(tab, text='Overview')

        toolbar = tk.Frame(tab, bd=1, relief=tk.RAISED)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        chk_pred_visible = tk.Checkbutton(
            toolbar, text='Show prediction', variable=self.var_pred_visible,
            command=lambda: self.update_current_tab())
        chk_pred_visible.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)
        if self.model is None:
            self.var_pred_visible.set(0)
            chk_pred_visible.configure(state='disabled')
        chk_gt_visible = tk.Checkbutton(
            toolbar, text='Show ground truth', variable=self.var_gt_visible,
            command=lambda: self.update_current_tab())
        if hasattr(self.dataset, 'subset') and self.dataset.subset == 'test':
            self.var_gt_visible.set(0)
            chk_gt_visible.configure(state='disabled')
        chk_gt_visible.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)

        fig = Figure()
        fig.subplots_adjust(0.05, 0.10, 0.95, 0.95, 0.05, 0.05)
        canvas = FigureCanvasTkAgg(fig, tab)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        nav_toolbar = NavigationToolbar2Tk(canvas, tab)
        nav_toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        prev_ax1: Axes3D = None

        def update_tab():
            fig.clf()

            skels = []
            if self.pred_visible:
                skels.append(self.current_example['pred_skel'])
            if self.gt_visible:
                skels.append(self.current_example['gt_skel'])

            ax1: Axes3D = fig.add_subplot(1, 2, 1, projection='3d')
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.imshow(self.current_example['input_image'])

            ground_truth = root_relative(self.current_example['gt_skel']['original'])
            for i, skel in enumerate(skels):
                alpha = 1 / (3 ** i)
                skel3d = root_relative(skel['camera_space'])
                if self.is_aligned:
                    skel3d = apply_rigid_alignment(skel3d, ground_truth)
                plot_skeleton_on_axes3d(skel3d, CanonicalSkeletonDesc,
                                        ax1, invert=True, alpha=alpha)
                plot_skeleton_on_axes(skel['image_space'], CanonicalSkeletonDesc, ax2, alpha=alpha)

            # Preserve 3D axes view
            nonlocal prev_ax1
            if prev_ax1 is not None:
                ax1.view_init(prev_ax1.elev, prev_ax1.azim)
            prev_ax1 = ax1

            canvas.draw()

        return update_tab

    def _make_heatmap_tab(self, notebook: ttk.Notebook):
        tab = tk.Frame(notebook)
        tab_index = len(notebook.tabs())
        notebook.add(tab, text='Heatmaps')

        if self.model is None:
            notebook.tab(tab_index, state='disabled')

        toolbar = tk.Frame(tab, bd=1, relief=tk.RAISED)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        joint_names = list(sorted(self.dataset.skeleton_desc.joint_names))

        opt_joint = tk.OptionMenu(
            toolbar, self.var_joint, *joint_names,
            command=lambda event: self.update_current_tab())
        opt_joint.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)

        var_image_visible = tk.IntVar(value=1)
        chk_image_visible = tk.Checkbutton(
            toolbar, text='Show image overlay', variable=var_image_visible,
            command=lambda: self.update_current_tab())
        chk_image_visible.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)

        var_mean_crosshairs = tk.IntVar(value=1)
        chk_mean_crosshairs = tk.Checkbutton(
            toolbar, text='Show mean', variable=var_mean_crosshairs,
            command=lambda: self.update_current_tab())
        chk_mean_crosshairs.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)

        fig = Figure()
        canvas = FigureCanvasTkAgg(fig, tab)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        nav_toolbar = NavigationToolbar2Tk(canvas, tab)
        nav_toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        prev_ax3d: Axes3D = None

        def update_tab():
            fig.clf()
            joint_index = self.dataset.skeleton_desc.joint_names.index(self.var_joint.get())

            cmap = plt.get_cmap('gist_yarg')
            img = self.current_example['input_image']
            hms = [
                (3, self.current_example['xy_heatmaps'][-1][joint_index], ('x', 'y')),
                (1, self.current_example['xz_heatmaps'][-1][joint_index], ('x', 'z')),
                (4, self.current_example['zy_heatmaps'][-1][joint_index], ('z', 'y')),
            ]

            for subplot_id, hm, (xlabel, ylabel) in hms:
                ax = fig.add_subplot(2, 2, subplot_id)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                extent = [0, hm.size(-1), hm.size(-2), 0]
                ax.imshow(hm, cmap=cmap, extent=extent)
                if subplot_id == 3 and var_image_visible.get() != 0:
                    ax.imshow(img, extent=extent, alpha=0.5)
                if var_mean_crosshairs.get() != 0:
                    ax.axvline(
                        np.average(np.arange(0, hm.size(-1)), weights=np.array(hm.sum(-2))),
                        ls='dashed',
                    )
                    ax.axhline(
                        np.average(np.arange(0, hm.size(-2)), weights=np.array(hm.sum(-1))),
                        ls='dashed',
                    )

            size = self.current_example['xy_heatmaps'][-1].size(-1)
            ax: Axes3D = fig.add_subplot(2, 2, 2, projection='3d')
            plot_skeleton_on_axes3d(
                (root_relative(self.current_example['pred_skel']['normalised']) + 1) * 0.5 * size,
                self.dataset.skeleton_desc, ax, invert=True)
            ax.set_xlim(0, size)
            ax.set_ylim(0, size)
            ax.set_zlim(size, 0)
            # Preserve 3D axes view
            nonlocal prev_ax3d
            if prev_ax3d is not None:
                ax.view_init(prev_ax3d.elev, prev_ax3d.azim)
            prev_ax3d = ax

            canvas.draw()

        return update_tab

    def on_change_example(self):
        self.current_example = load_and_process_example(
            self.dataset, self.current_example_index, self.device, self.model)

        self.update_current_tab()


def main(argv, common_opts):
    args = parse_args(argv)
    seed_all(12345)
    init_algorithms(deterministic=True)
    torch.set_grad_enabled(False)

    device = common_opts['device']

    if args.model:
        model = load_model(args.model).to(device).eval()
        data_specs = model.data_specs
    else:
        model = None
        data_specs = DataSpecs(
            ImageSpecs(224, mean=ImageSpecs.IMAGENET_MEAN, stddev=ImageSpecs.IMAGENET_STDDEV),
            JointsSpecs(CanonicalSkeletonDesc, n_dims=3),
        )

    dataset = get_dataset(args.dataset, data_specs, use_aug=False)

    app = MainGUIApp(dataset, device, model)
    app.mainloop()


GUI_Subcommand = Subcommand(name='gui', func=main, help='browse examples and predictions')

if __name__ == '__main__':
    GUI_Subcommand.run()
