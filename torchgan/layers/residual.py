import torch.nn as nn

__all__ = ['ResidualBlock']

class ResidualBlock(nn.Module):
    def __init__(self, filters, kernels, strides=None, paddings=None, activation=nn.ReLU,
                 batchnorm=True, shortcut=None):
        super(ResidualBlock, self).__init__()
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
                layers.append(activation())
        self.layers = nn.Sequential(*layers)
        self.shortcut = shortcut

    def forward(self, x, activation=None):
        out = self.layers(x)
        if self.shortcut is not None:
            out += self.shortcut(x)
        if activation is None:
            return out
        else:
            return activation(out)
