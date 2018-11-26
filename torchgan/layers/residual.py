import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, filters, kernels, paddings, strides, activation=nn.ReLU,
                 batchnorm=True, shortcut=None):
        super(ResidualBlock, self).__init__()
        assert len(filters) == len(kernels) + 1 and len(filters) == len(strides) + 1 \
            and len(filters) == len(paddings) + 1
        layers = []
        for i in range(1, len(filters)):
            layers.append(nn.Conv2d(filters[i - 1], filters[i], kernels[i], strides[i], paddings[i]))
            if batchnorm:
                layers.append(nn.BatchNorm2d(filters[i]))
            if i != len(filters): # Last layer does not get an activation
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
