import torch
from torch import nn
import torchvision.models
from pretrainedmodels.models.inceptionv4 import inceptionv4
from margipose.dsntnn import flat_softmax, dsnt, js_reg_losses, euclidean_losses
from semantic_version import Version, Spec

from margipose.model_factory import ModelFactory
from margipose.nn_helpers import init_parameters
from margipose.data.skeleton import CanonicalSkeletonDesc
from margipose.data_specs import DataSpecs, ImageSpecs, JointsSpecs


class ResidualBlock(nn.Module):
    def __init__(self, module, shortcut=None):
        super().__init__()
        self.module = module
        if shortcut is None:
            shortcut = lambda x: x
        self.shortcut = shortcut

    def forward(self, *inputs):
        return self.module(inputs[0]) + self.shortcut(inputs[0])


class ConvFactory():
    def __init__(self, disable_dilation):
        self.d = 1
        if disable_dilation:
            self.dilation_factor = 1
        else:
            self.dilation_factor = 2

    def conv3x3(self, in_chans, out_chans):
        return nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=self.d, dilation=self.d,
                         bias=False)

    def down_dilate(self, in_chans, out_chans):
        module = self.conv3x3(in_chans, out_chans)
        self.d *= self.dilation_factor
        return module

    def down_stride(self, in_chans, out_chans):
        return nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=self.d, stride=2,
                         dilation=self.d, bias=False)

    def down_dilate_block(self, in_chans, out_chans):
        module = nn.Sequential(
            self.down_dilate(in_chans, out_chans),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
            self.conv3x3(out_chans, out_chans),
             nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
        )
        shortcut = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_chans),
        )
        return ResidualBlock(module, shortcut)

    def down_stride_block(self, in_chans, out_chans):
        module = nn.Sequential(
            self.down_stride(in_chans, out_chans),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
            self.conv3x3(out_chans, out_chans),
             nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
        )
        shortcut = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_chans),
        )
        return ResidualBlock(module, shortcut)

    def up_dilate(self, in_chans, out_chans):
        module = self.conv3x3(in_chans, out_chans)
        self.d //= self.dilation_factor
        return module

    def up_stride(self, in_chans, out_chans):
        return nn.ConvTranspose2d(in_chans, out_chans, kernel_size=3, padding=self.d, stride=2,
                                  output_padding=1, dilation=self.d, bias=False)

    def up_dilate_block(self, in_chans, out_chans):
        module = nn.Sequential(
            self.up_dilate(in_chans, out_chans),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
            self.conv3x3(out_chans, out_chans),
             nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
        )
        shortcut = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_chans),
        )
        return ResidualBlock(module, shortcut)

    def up_stride_block(self, in_chans, out_chans):
        module = nn.Sequential(
            self.up_stride(in_chans, out_chans),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
            self.conv3x3(out_chans, out_chans),
             nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
        )
        shortcut = nn.Sequential(
            nn.ConvTranspose2d(in_chans, out_chans, kernel_size=1, stride=2, output_padding=1, bias=False),
            nn.BatchNorm2d(out_chans),
        )
        return ResidualBlock(module, shortcut)


class HeatmapColumn(nn.Module):
    def __init__(self, n_joints, heatmap_space, disable_dilation):
        super().__init__()
        self.n_joints = n_joints
        self.heatmap_space = heatmap_space
        cf = ConvFactory(disable_dilation)
        self.down_layers = nn.Sequential(
            cf.down_dilate_block(128, 128),
            cf.down_dilate_block(128, 128),
            cf.down_stride_block(128, 192),
            cf.down_dilate_block(192, 192),
            cf.down_dilate_block(192, 192),
        )
        self.up_layers = nn.Sequential(
            cf.up_dilate_block(192, 192),
            cf.up_dilate_block(192, 192),
            cf.up_stride_block(192, 128),
            cf.up_dilate_block(128, 128),
            cf.up_dilate_block(128, self.n_joints),
        )
        init_parameters(self)

    def forward(self, *inputs):
        mid_in = self.down_layers(inputs[0])
        # Spatial size (width = height = depth). Must divide evenly into # channels
        size = mid_in.size(-1)
        if self.heatmap_space == 'xy':
            mid_out = mid_in
        elif self.heatmap_space == 'zy':
            mid_out = torch.cat(
                [t.permute(0, 3, 2, 1) for t in mid_in.split(size, -3)],
            -3)
        elif self.heatmap_space == 'xz':
            mid_out = torch.cat(
                [t.permute(0, 2, 1, 3) for t in mid_in.split(size, -3)],
            -3)
        else:
            raise Exception()
        return self.up_layers(mid_out)


def make_image_feature_extractor(model_name):
    if model_name == 'inceptionv4':
        net = nn.Sequential(
            *[inceptionv4().features[i] for i in range(7)],
            nn.Conv2d(384, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        for module in net.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.MaxPool2d):
                kernel_size = module.kernel_size
                if isinstance(kernel_size, int):
                    module.padding = kernel_size // 2
                else:
                    module.padding = tuple([k // 2 for k in kernel_size])
        return net
    elif model_name in {'resnet18', 'resnet34', 'resnet50'}:
        resnet = getattr(torchvision.models, model_name)(pretrained=True)
        net = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
        )
        return net
    raise Exception('unsupported image feature extractor model name: ' + model_name)


class HeatmapCombiner(nn.Module):
    def __init__(self, n_joints):
        super().__init__()
        self.conv = nn.Conv2d(n_joints * 3, 128, kernel_size=1, bias=False)
        init_parameters(self)

    def forward(self, *inputs):
        xy_hm, zy_hm, xz_hm = inputs
        return self.conv(torch.cat([xy_hm, zy_hm, xz_hm], -3))


class MargiPoseModelInner(nn.Module):
    def __init__(self, n_joints, n_stages, bad_permutation, disable_permutation,
                 feature_extractor, disable_dilation):
        super().__init__()

        self.n_stages = n_stages
        self.in_cnn = make_image_feature_extractor(feature_extractor)
        self.xy_hm_cnns = nn.ModuleList()
        self.zy_hm_cnns = nn.ModuleList()
        self.xz_hm_cnns = nn.ModuleList()
        self.hm_combiners = nn.ModuleList()

        xy = 'xy'
        if disable_permutation:
            zy = 'xy'
            xz = 'xy'
        elif bad_permutation:
            zy = 'xz'
            xz = 'zy'
        else:
            zy = 'zy'
            xz = 'xz'

        for t in range(self.n_stages):
            if t > 0:
                self.hm_combiners.append(HeatmapCombiner(n_joints))
            self.xy_hm_cnns.append(HeatmapColumn(n_joints, heatmap_space=xy, disable_dilation=disable_dilation))
            self.zy_hm_cnns.append(HeatmapColumn(n_joints, heatmap_space=zy, disable_dilation=disable_dilation))
            self.xz_hm_cnns.append(HeatmapColumn(n_joints, heatmap_space=xz, disable_dilation=disable_dilation))

    def forward(self, *inputs):
        features = self.in_cnn(inputs[0])

        # These lists will store the outputs from each stage
        xy_heatmaps = []
        zy_heatmaps = []
        xz_heatmaps = []

        inp = features
        for t in range(self.n_stages):
            if t > 0:
                combined_hm_features = self.hm_combiners[t - 1](
                    xy_heatmaps[t - 1],
                    zy_heatmaps[t - 1],
                    xz_heatmaps[t - 1],
                )
                inp = inp + combined_hm_features
            xy_heatmaps.append(flat_softmax(self.xy_hm_cnns[t](inp)))
            zy_heatmaps.append(flat_softmax(self.zy_hm_cnns[t](inp)))
            xz_heatmaps.append(flat_softmax(self.xz_hm_cnns[t](inp)))

        return (xy_heatmaps, zy_heatmaps, xz_heatmaps)


class MargiPoseModel(nn.Module):
    def __init__(self, skel_desc, only_2d=False, coord_space='ndc', n_stages=4,
                 bad_permutation=False, disable_permutation=False, data_parallel=False,
                 feature_extractor='inceptionv4', average_xy=False, disable_dilation=False,
                 disable_reg=False):
        super().__init__()

        self.data_specs = DataSpecs(
            ImageSpecs(256, mean=ImageSpecs.IMAGENET_MEAN, stddev=ImageSpecs.IMAGENET_STDDEV),
            JointsSpecs(skel_desc, n_dims=3, coord_space=coord_space),
        )
        self.only_2d = only_2d
        self.average_xy = average_xy
        self.disable_reg = disable_reg

        self.inner = MargiPoseModelInner(skel_desc.n_joints, n_stages, bad_permutation,
                                         disable_permutation, feature_extractor, disable_dilation)
        if data_parallel:
            self.inner = nn.DataParallel(self.inner)

    def _sigma(self):
        return 1.0

    def forward_2d_losses(self, out_var, target_var):
        sigma = self._sigma()

        target_xy = target_var.narrow(-1, 0, 2)
        losses = 0

        for xy_hm, zy_hm, xz_hm in zip(self.xy_heatmaps, self.zy_heatmaps, self.xz_heatmaps):
            if not self.disable_reg:
                losses += js_reg_losses(xy_hm, target_xy, sigma)
            pred_xy = self.heatmaps_to_coords(xy_hm, zy_hm, xz_hm).narrow(-1, 0, 2)
            losses += euclidean_losses(pred_xy, target_xy)

        return losses

    def forward_3d_losses(self, out_var, target_var):
        sigma = self._sigma()

        target_xyz = target_var.narrow(-1, 0, 3)
        losses = 0

        target_xy = target_xyz.narrow(-1, 0, 2)
        target_zy = torch.cat([target_xyz.narrow(-1, 2, 1), target_xyz.narrow(-1, 1, 1)], -1)
        target_xz = torch.cat([target_xyz.narrow(-1, 0, 1), target_xyz.narrow(-1, 2, 1)], -1)
        for xy_hm, zy_hm, xz_hm in zip(self.xy_heatmaps, self.zy_heatmaps, self.xz_heatmaps):
            if not self.disable_reg:
                losses += js_reg_losses(xy_hm, target_xy, sigma)
                losses += js_reg_losses(zy_hm, target_zy, sigma)
                losses += js_reg_losses(xz_hm, target_xz, sigma)
            pred_xyz = self.heatmaps_to_coords(xy_hm, zy_hm, xz_hm)
            losses += euclidean_losses(pred_xyz, target_xyz)

        return losses

    def heatmaps_to_coords(self, xy_hm, zy_hm, xz_hm):
        xy = dsnt(xy_hm)
        zy = dsnt(zy_hm)
        xz = dsnt(xz_hm)
        if self.average_xy:
            x = 0.5 * (xy[:, :, 0:1] + xz[:, :, 0:1])
            y = 0.5 * (xy[:, :, 1:2] + zy[:, :, 1:2])
        else:
            x, y = xy.split(1, -1)
        z = 0.5 * (zy[:, :, 0:1] + xz[:, :, 1:2])
        return torch.cat([x, y, z], -1)

    def forward(self, *inputs):
        self.xy_heatmaps, self.zy_heatmaps, self.xz_heatmaps = self.inner(*inputs)
        xyz = self.heatmaps_to_coords(self.xy_heatmaps[-1], self.zy_heatmaps[-1], self.xz_heatmaps[-1])
        if self.only_2d:
            return xyz.narrow(-1, 0, 2)
        return xyz


class OldMargiPoseModelFactory(ModelFactory):
    def __init__(self,):
        super().__init__('margipose', '^4.0.0,>=4.2.0')

    def create(self, model_desc):
        super()
        settings = model_desc['settings']
        version = Version(model_desc['version'])
        s = dict(
            coord_space=settings.get('coord_space', 'ndc'),
            n_stages=settings.get('n_stages', 4),
            bad_permutation=settings.get('bad_permutation', False),
            disable_permutation=settings.get('disable_permutation', False),
            data_parallel=settings.get('data_parallel', False),
            feature_extractor=settings.get('feature_extractor', 'inceptionv4'),
            average_xy=settings.get('average_xy', False),
            disable_reg=settings.get('disable_reg', False),
            disable_dilation=settings.get('disable_dilation', version in Spec('>=4.2.4'))
        )
        return MargiPoseModel(CanonicalSkeletonDesc, only_2d=False, **s)
