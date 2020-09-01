import torch
from torch import nn
import torchvision.models
from pretrainedmodels.models.inceptionv4 import inceptionv4
from margipose.dsntnn import flat_softmax, dsnt, js_reg_losses, euclidean_losses

from margipose.model_factory import ModelFactory
from margipose.nn_helpers import init_parameters
from margipose.data.skeleton import CanonicalSkeletonDesc
from margipose.data_specs import DataSpecs, ImageSpecs, JointsSpecs


Default_MargiPose_Desc = {
    'type': 'margipose',
    'version': '6.0.1',
    'settings': {
        'n_stages': 4,
        'axis_permutation': True,
        'feature_extractor': 'inceptionv4',
        'pixelwise_loss': 'jsd',
    },
}


class ResidualBlock(nn.Module):
    def __init__(self, chans, main_conv_in, shortcut_conv_in):
        super().__init__()
        assert main_conv_in.in_channels == shortcut_conv_in.in_channels
        self.module = nn.Sequential(
            main_conv_in,
            nn.BatchNorm2d(chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(chans, chans, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(chans),
            nn.ReLU(inplace=True),
        )
        self.shortcut = nn.Sequential(shortcut_conv_in, nn.BatchNorm2d(chans))

    def forward(self, *inputs):
        return self.module(inputs[0]) + self.shortcut(inputs[0])


class HeatmapColumn(nn.Module):
    def __init__(self, n_joints, heatmap_space):
        super().__init__()
        self.n_joints = n_joints
        self.heatmap_space = heatmap_space
        self.down_layers = nn.Sequential(
            self._regular_block(128, 128),
            self._regular_block(128, 128),
            self._down_stride_block(128, 192),
            self._regular_block(192, 192),
            self._regular_block(192, 192),
        )
        self.up_layers = nn.Sequential(
            self._regular_block(192, 192),
            self._regular_block(192, 192),
            self._up_stride_block(192, 128),
            self._regular_block(128, 128),
            self._regular_block(128, self.n_joints),
        )
        init_parameters(self)

    def _regular_block(self, in_chans, out_chans):
        return ResidualBlock(
            out_chans,
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=False))

    def _down_stride_block(self, in_chans, out_chans):
        return ResidualBlock(
            out_chans,
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, stride=2, bias=False),
            nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=2, bias=False))

    def _up_stride_block(self, in_chans, out_chans):
        return ResidualBlock(
            out_chans,
            nn.ConvTranspose2d(in_chans, out_chans, kernel_size=3, padding=1, stride=2,
                               output_padding=1, bias=False),
            nn.ConvTranspose2d(in_chans, out_chans, kernel_size=1, stride=2,
                               output_padding=1, bias=False))

    def forward(self, *inputs):
        mid_in = self.down_layers(inputs[0])
        # Spatial size (width = height = depth). Must divide evenly into # channels
        # FIXME: The conversion to int generates a warning during tracing, and is not necessary
        #        since PyTorch 1.5.0. However, it is required for tracing to work at all with prior
        #        PyTorch versions. When we upgrade to PyTorch 1.5.0, we should change this line.
        #        See: https://github.com/pytorch/pytorch/issues/27551
        size = int(mid_in.shape[-1])
        if self.heatmap_space == 'xy':
            mid_out = mid_in
        elif self.heatmap_space == 'zy':
            mid_out = torch.cat([t.permute(0, 3, 2, 1) for t in mid_in.split(size, -3)], -3)
        elif self.heatmap_space == 'xz':
            mid_out = torch.cat([t.permute(0, 2, 1, 3) for t in mid_in.split(size, -3)], -3)
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
        extra_modules = []
        resnet_out_chans = resnet.layer3[0].conv1.in_channels
        if resnet_out_chans != 128:
            extra_modules = [
                nn.Conv2d(resnet_out_chans, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            ]
        net = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            *extra_modules,
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
    def __init__(self, n_joints, n_stages, axis_permutation, feature_extractor):
        super().__init__()

        self.n_stages = n_stages
        self.in_cnn = make_image_feature_extractor(feature_extractor)
        self.xy_hm_cnns = nn.ModuleList()
        self.zy_hm_cnns = nn.ModuleList()
        self.xz_hm_cnns = nn.ModuleList()
        self.hm_combiners = nn.ModuleList()

        xy = 'xy'
        if axis_permutation:
            zy = 'zy'
            xz = 'xz'
        else:
            zy = 'xy'
            xz = 'xy'

        for t in range(self.n_stages):
            if t > 0:
                self.hm_combiners.append(HeatmapCombiner(n_joints))
            self.xy_hm_cnns.append(HeatmapColumn(n_joints, heatmap_space=xy))
            self.zy_hm_cnns.append(HeatmapColumn(n_joints, heatmap_space=zy))
            self.xz_hm_cnns.append(HeatmapColumn(n_joints, heatmap_space=xz))

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

        return xy_heatmaps, zy_heatmaps, xz_heatmaps


class MargiPoseModel(nn.Module):
    def __init__(self, skel_desc, n_stages, axis_permutation, feature_extractor, pixelwise_loss):
        super().__init__()
        self.data_specs = DataSpecs(
            ImageSpecs(256, mean=ImageSpecs.IMAGENET_MEAN, stddev=ImageSpecs.IMAGENET_STDDEV),
            JointsSpecs(skel_desc, n_dims=3),
        )
        self.pixelwise_loss = pixelwise_loss
        self.inner = MargiPoseModelInner(skel_desc.n_joints, n_stages, axis_permutation,
                                         feature_extractor)
        self.xy_heatmaps = self.zy_heatmaps = self.xz_heatmaps = None

    def _calculate_pixelwise_loss(self, hm, target_coords):
        sigma = 1.0
        if self.pixelwise_loss == 'jsd':
            return js_reg_losses(hm, target_coords, sigma)
        elif self.pixelwise_loss is None:
            return 0
        raise Exception('unrecognised pixelwise loss: {}'.format(self.pixelwise_loss))

    def forward_2d_losses(self, out_var, target_var):
        target_xy = target_var.narrow(-1, 0, 2)
        losses = 0

        for xy_hm, zy_hm, xz_hm in zip(self.xy_heatmaps, self.zy_heatmaps, self.xz_heatmaps):
            # Pixelwise heatmap loss.
            losses += self._calculate_pixelwise_loss(xy_hm, target_xy)
            # Calculated coordinate loss.
            actual_xy = self.heatmaps_to_coords(xy_hm, zy_hm, xz_hm)[..., :2]
            losses += euclidean_losses(actual_xy, target_xy)

        return losses

    def forward_3d_losses(self, out_var, target_var):
        target_xyz = target_var.narrow(-1, 0, 3)
        losses = 0

        target_xy = target_xyz.narrow(-1, 0, 2)
        target_zy = torch.cat([target_xyz.narrow(-1, 2, 1), target_xyz.narrow(-1, 1, 1)], -1)
        target_xz = torch.cat([target_xyz.narrow(-1, 0, 1), target_xyz.narrow(-1, 2, 1)], -1)
        for xy_hm, zy_hm, xz_hm in zip(self.xy_heatmaps, self.zy_heatmaps, self.xz_heatmaps):
            # Pixelwise heatmap loss.
            losses += self._calculate_pixelwise_loss(xy_hm, target_xy)
            losses += self._calculate_pixelwise_loss(zy_hm, target_zy)
            losses += self._calculate_pixelwise_loss(xz_hm, target_xz)
            # Calculated coordinate loss.
            actual_xyz = self.heatmaps_to_coords(xy_hm, zy_hm, xz_hm)
            losses += euclidean_losses(actual_xyz, target_xyz)

        return losses

    @staticmethod
    def heatmaps_to_coords(xy_hm, zy_hm, xz_hm):
        xy = dsnt(xy_hm)
        zy = dsnt(zy_hm)
        xz = dsnt(xz_hm)
        x, y = xy.split(1, -1)
        z = 0.5 * (zy[:, :, 0:1] + xz[:, :, 1:2])
        return torch.cat([x, y, z], -1)

    def forward(self, *inputs):
        self.xy_heatmaps, self.zy_heatmaps, self.xz_heatmaps = self.inner(*inputs)
        xyz = self.heatmaps_to_coords(self.xy_heatmaps[-1], self.zy_heatmaps[-1],
                                      self.xz_heatmaps[-1])
        return xyz


class MargiPoseModelFactory(ModelFactory):
    def __init__(self,):
        super().__init__('margipose', '^6.0.0')

    def create(self, model_desc):
        super()
        s = model_desc['settings']
        kwargs = dict(
            skel_desc=CanonicalSkeletonDesc,
            n_stages=s.get('n_stages', 4),
            axis_permutation=s.get('axis_permutation', True),
            feature_extractor=s.get('feature_extractor', 'inceptionv4'),
            pixelwise_loss=s.get('pixelwise_loss', 'jsd'),
        )
        return MargiPoseModel(**kwargs)
