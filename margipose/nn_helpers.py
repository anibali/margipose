from math import sqrt
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn import init


def init_parameters(net):
    for m in net.modules():
        if isinstance(m, _ConvNd):
            init.kaiming_normal(m.weight, 0, 'fan_out')
            if m.bias is not None:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # Kaiming initialisation for linear layers
            init.normal(m.weight, 0, sqrt(2.0 / m.weight.size(0)))
            if m.bias is not None:
                init.normal(m.bias, 0, sqrt(2.0 / m.bias.size(0)))
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            if m.bias is not None:
                init.constant(m.bias, 0)
