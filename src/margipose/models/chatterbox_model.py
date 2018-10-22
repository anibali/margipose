import torch
from torch import nn
from torch.nn.functional import relu, max_pool2d
from torchvision.models.resnet import BasicBlock, resnet34

from margipose.data.skeleton import CanonicalSkeletonDesc
from margipose.data_specs import DataSpecs, ImageSpecs, JointsSpecs
from margipose.dsntnn import flat_softmax, dsnt, js_reg_losses, euclidean_losses
from margipose.model_factory import ModelFactory
from margipose.nn_helpers import init_parameters


Default_Chatterbox_Desc = {
    'type': 'chatterbox',
    'version': '1.3.0',
    'settings': {
        'pixelwise_loss': 'jsd',
    },
}


def _make_block_group(in_planes, out_planes, n_blocks, stride=1):
    downsample = None
    if stride != 1 or in_planes != out_planes:
        downsample = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_planes),
        )

    layers = [BasicBlock(in_planes, out_planes, stride, downsample)]
    layers += [BasicBlock(out_planes, out_planes) for _ in range(n_blocks)]

    return nn.Sequential(*layers)


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, resnet):
        super().__init__()

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2

    def forward(self, *inputs):
        x = inputs[0]
        x = self.conv1(x)
        x = self.bn1(x)
        x = relu(x, True)
        x = max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class _XYCnn(nn.Module):
    def __init__(self, resnet, n_joints):
        super().__init__()

        layers = [resnet.layer3, resnet.layer4]
        for i, layer in enumerate(layers):
            dilx = dily = 2 ** (i + 1)
            for module in layer.modules():
                if isinstance(module, nn.Conv2d):
                    if module.stride == (2, 2):
                        module.stride = (1, 1)
                    elif module.kernel_size == (3, 3):
                        kx, ky = module.kernel_size
                        module.dilation = (dilx, dily)
                        module.padding = ((dilx * (kx - 1) + 1) // 2, (dily * (ky - 1) + 1) // 2)
        self.layer1, self.layer2 = layers

        self.hm_conv = nn.Conv2d(512, n_joints, kernel_size=1, bias=False)
        init_parameters(self.hm_conv)

    def forward(self, *inputs):
        t = inputs[0]

        t = self.layer1(t)
        t = self.layer2(t)
        t = self.hm_conv(t)

        return t


class _ChatterboxCnn(nn.Module):
    def __init__(self, n_joints, shrink_width=True):
        super().__init__()

        def f(a, b):
            if shrink_width:
                return (a, b)
            else:
                return (b, a)

        self.down_convs = nn.Sequential(
            # 128 x 32 x 32
            self._DownBlock(128, 256, stride=f(1, 2), dilation=f(2, 1), dilation_in=f(1, 1)),
            self._DownBlock(256, 256, dilation=f(2, 1)),
            # 256 x 32 x 16
            self._DownBlock(256, 512, stride=f(1, 2), dilation=f(4, 1), dilation_in=f(2, 1)),
            self._DownBlock(512, 512, dilation=f(4, 1)),
            # 512 x 32 x 8
            nn.Conv2d(512, 1024, kernel_size=f(1, 8), bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            # 1024 x 32 x 1
        )

        self.up_convs = nn.Sequential(
            # 1024 x 32 x 1
            nn.ConvTranspose2d(1024, 512, kernel_size=f(1, 8), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 512 x 32 x 8
            self._UpBlock(512, 512, dilation=f(4, 1)),
            self._UpBlock(512, 256, stride=f(1, 2), dilation=f(2, 1), dilation_in=f(4, 1),
                          output_padding=f(0, 1)),
            # 256 x 32 x 16
            self._UpBlock(256, 256, dilation=f(2, 1)),
            self._UpBlock(256, 128, stride=f(1, 2), dilation=f(1, 1), dilation_in=f(2, 1),
                          output_padding=f(0, 1)),
            # 128 x 32 x 32
            nn.Conv2d(128, n_joints, kernel_size=1, bias=False),
            # n x 32 x 32
        )

        init_parameters(self)

    class _DownBlock(nn.Module):
        def __init__(self, in_planes, out_planes, stride=1, dilation=(1, 1), dilation_in=None):
            super().__init__()

            if dilation_in is None:
                dilation_in = dilation

            if stride != 1 or in_planes != out_planes:
                self.resample = nn.Sequential(
                    nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_planes),
                )
            else:
                self.resample = None

            self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride=stride,
                                   padding=dilation_in, dilation=dilation_in, bias=False)
            self.bn1 = nn.BatchNorm2d(out_planes)
            self.conv2 = nn.Conv2d(out_planes, out_planes, 3, padding=dilation,
                                   dilation=dilation, bias=False)
            self.bn2 = nn.BatchNorm2d(out_planes)

        def forward(self, x):
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = relu(out, True)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.resample is not None:
                residual = self.resample(x)

            out += residual
            out = relu(out, True)

            return out

    class _UpBlock(nn.Module):
        def __init__(self, in_planes, out_planes, stride=1, dilation=(1, 1), dilation_in=None,
                     output_padding=(0, 0)):
            super().__init__()

            if dilation_in is None:
                dilation_in = dilation

            if stride != 1 or in_planes != out_planes:
                self.resample = nn.Sequential(
                    nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                       output_padding=output_padding, bias=False),
                    nn.BatchNorm2d(out_planes),
                )
            else:
                self.resample = None

            self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, 3, stride=stride,
                                            padding=dilation_in, dilation=dilation_in,
                                            output_padding=output_padding, bias=False)
            self.bn1 = nn.BatchNorm2d(out_planes)
            self.conv2 = nn.Conv2d(out_planes, out_planes, 3, padding=dilation,
                                   dilation=dilation, bias=False)
            self.bn2 = nn.BatchNorm2d(out_planes)

        def forward(self, x):
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = relu(out, True)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.resample is not None:
                residual = self.resample(x)

            out += residual
            out = relu(out, True)

            return out

    def forward(self, *inputs):
        t = inputs[0]

        t = self.down_convs(t)
        # At this point, t should have one of width/height set to 1
        t = self.up_convs(t)

        return t


class ChatterboxModel(nn.Module):
    def __init__(self, skel_desc, pixelwise_loss):
        super().__init__()

        self.data_specs = DataSpecs(
            ImageSpecs(256, mean=ImageSpecs.IMAGENET_MEAN, stddev=ImageSpecs.IMAGENET_STDDEV),
            JointsSpecs(skel_desc, n_dims=3),
        )

        self.pixelwise_loss = pixelwise_loss

        resnet = resnet34(pretrained=True)
        self.in_cnn = ResNetFeatureExtractor(resnet)
        self.xy_hm_cnn = _XYCnn(resnet, skel_desc.n_joints)

        self.zy_hm_cnn = _ChatterboxCnn(skel_desc.n_joints, shrink_width=True)
        self.xz_hm_cnn = _ChatterboxCnn(skel_desc.n_joints, shrink_width=False)

    def _calculate_pixelwise_loss(self, hm, target_coords):
        sigma = 1.0
        if self.pixelwise_loss == 'jsd':
            return js_reg_losses(hm, target_coords, sigma)
        elif self.pixelwise_loss is None:
            return 0
        raise Exception('unrecognised pixelwise loss: {}'.format(self.pixelwise_loss))

    def forward_2d_losses(self, out_var, target_var):
        out_xy = out_var.narrow(-1, 0, 2)
        target_xy = target_var.narrow(-1, 0, 2)

        losses = (euclidean_losses(out_xy, target_xy) +
                  self._calculate_pixelwise_loss(self.xy_heatmaps[-1], target_xy))

        return losses

    def forward_3d_losses(self, out_var, target_var):
        out_xyz = out_var.narrow(-1, 0, 3)
        target_xyz = target_var.narrow(-1, 0, 3)

        target_xy = target_xyz.narrow(-1, 0, 2)
        target_zy = torch.cat([target_xyz.narrow(-1, 2, 1), target_xyz.narrow(-1, 1, 1)], -1)
        target_xz = torch.cat([target_xyz.narrow(-1, 0, 1), target_xyz.narrow(-1, 2, 1)], -1)

        losses = (euclidean_losses(out_xyz, target_xyz) +
                  self._calculate_pixelwise_loss(self.xy_heatmaps[-1], target_xy) +
                  self._calculate_pixelwise_loss(self.zy_heatmaps[-1], target_zy) +
                  self._calculate_pixelwise_loss(self.xz_heatmaps[-1], target_xz))

        return losses

    def forward(self, *inputs):
        t = inputs[0]
        t = self.in_cnn(t)

        self.xy_heatmaps = [flat_softmax(self.xy_hm_cnn(t))]
        self.zy_heatmaps = [flat_softmax(self.zy_hm_cnn(t))]
        self.xz_heatmaps = [flat_softmax(self.xz_hm_cnn(t))]

        xy = dsnt(self.xy_heatmaps[-1])
        zy = dsnt(self.zy_heatmaps[-1])
        xz = dsnt(self.xz_heatmaps[-1])

        x = xy.narrow(-1, 0, 1)
        y = xy.narrow(-1, 1, 1)
        z = 0.5 * (zy.narrow(-1, 0, 1) + xz.narrow(-1, 1, 1))

        return torch.cat([x, y, z], -1)


class ChatterboxModelFactory(ModelFactory):
    def __init__(self,):
        super().__init__('chatterbox', '^1.3.0')

    def create(self, model_desc):
        super()
        s = model_desc['settings']
        kwargs = dict(
            skel_desc=CanonicalSkeletonDesc,
            pixelwise_loss=s.get('pixelwise_loss', 'jsd'),
        )
        return ChatterboxModel(**kwargs)
