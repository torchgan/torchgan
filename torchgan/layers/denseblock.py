import torch
import torch.nn as nn

__all__ = ['BasicBlock', 'BottleneckBlock', 'TransitionBlock', 'TransitionBlockTranspose', 'DenseBlock']

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, batchnorm=True, nonlinearity=None):
        super(BasicBlock, self).__init__()
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        if batchnorm is True:
            self.model = nn.Sequential(
                nn.BatchNorm2d(in_channels), nl,
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
        else:
            self.model = nn.Sequential(nl, nn.Conv2d(in_channels, out_channels,
                kernel_size, stride, padding, bias=True))

    def forward(self, x):
        return torch.cat([x, self.model(x)], 1)


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck_channels,
            kernel_size, stride, padding, batchnorm=True, nonlinearity=None):
        super(BottleneckBlock, self).__init__()
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        if batchnorm is True:
            self.model = nn.Sequential(
                nn.BatchNorm2d(in_channels), nl,
                nn.Conv2d(in_channels, bottleneck_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(bottleneck_channels), nl,
                nn.Conv2d(bottleneck_channels, out_channels, kernel_size, stride, padding, bias=False))
        else:
            self.model = nn.Sequential(
                nl, nn.Conv2d(in_channels, bottleneck_channels, 1, 1, 0, bias=True),
                nl, nn.Conv2d(bottleneck_channels, out_channels, kernel_size, stride, padding, bias=True))

    def forward(self, x):
        return torch.cat([x, self.model(x)], 1)


class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, batchnorm=True, nonlinearity=None):
        super(TransitionBlock, self).__init__()
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        if batchnorm is True:
            self.model = nn.Sequential(
                nn.BatchNorm2d(in_channels), nl,
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
        else:
            self.model = nn.Sequential(nl, nn.Conv2d(in_channels, out_channels, kernel_size,
                stride, padding, bias=True))

    def forward(self, x):
        return self.model(x)


class TransitionBlockTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, batchnorm=True, nonlinearity=None):
        super(TransitionBlockTranspose, self).__init__()
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        if batchnorm is True:
            self.model = nn.Sequential(
                nn.BatchNorm2d(in_channels), nl,
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
        else:
            self.model = nn.Sequential(nl, nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                stride, padding, bias=True))

    def forward(self, x):
        return self.model(x)


class DenseBlock(nn.Module):
    def __init__(self, n, in_channels, growth_rate, block, batchnorm=True, nonlinearity=None):
        super(DenseBlock, self).__init__()
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        model = []
        for i in range(n):
            model.append(block(in_channels + i * growth_rate, growth_rate, nonlinearity=nl))
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
