import torch.nn as nn

__all__ = ['ResidualBlock2d', 'ResidualBlockTranspose2d']

class ResidualBlock2d(nn.Module):
    def __init__(self, filters, kernels, strides=None, paddings=None, nonlinearity=None,
                 batchnorm=True, shortcut=None):
        super(ResidualBlock2d, self).__init__()
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        if strides is None:
            strides = [1 for _ in range(len(kernels))]
        if paddings is None:
            paddings = [0 for _ in range(len(kernels))]
        assert len(filters) == len(kernels) + 1 and len(filters) == len(strides) + 1 \
            and len(filters) == len(paddings) + 1
        layers = []
        for i in range(1, len(filters)):
            layers.append(nn.Conv2d(filters[i - 1], filters[i], kernels[i - 1], strides[i - 1], paddings[i - 1]))
            if batchnorm:
                layers.append(nn.BatchNorm2d(filters[i]))
            if i != len(filters):  # Last layer does not get an activation
                layers.append(nl)
        self.layers = nn.Sequential(*layers)
        self.shortcut = shortcut

    def forward(self, x, nonlinearity=None):
        out = self.layers(x)
        if self.shortcut is not None:
            out += self.shortcut(x)
        return out if nonlinearity is None else nonlinearity(out)

class ResidualBlockTranspose2d(nn.Module):
    def __init__(self, filters, kernels, strides=None, paddings=None, nonlinearity=None,
                 batchnorm=True, shortcut=None):
        super(ResidualBlockTranspose2d, self).__init__()
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        if strides is None:
            strides = [1 for _ in range(len(kernels))]
        if paddings is None:
            paddings = [0 for _ in range(len(kernels))]
        assert len(filters) == len(kernels) + 1 and len(filters) == len(strides) + 1 \
            and len(filters) == len(paddings) + 1
        layers = []
        for i in range(1, len(filters)):
            layers.append(nn.ConvTranspose2d(filters[i - 1], filters[i], kernels[i - 1],
                                             strides[i - 1], paddings[i - 1]))
            if batchnorm:
                layers.append(nn.BatchNorm2d(filters[i]))
            if i != len(filters):  # Last layer does not get an activation
                layers.append(nl)
        self.layers = nn.Sequential(*layers)
        self.shortcut = shortcut

    def forward(self, x, nonlinearity=None):
        out = self.layers(x)
        if self.shortcut is not None:
            out += self.shortcut(x)
        return out if nonlinearity is None else nonlinearity(out)
