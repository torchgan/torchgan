import torch.nn as nn

__all__ = ['ResidualBlock2d']

class ResidualBlock2d(nn.Module):
    r"""Residual Block Module as described in `"Deep Residual Learning for Image Recognition
    by He et. al." <https://arxiv.org/abs/1512.03385>`_

    The output of the residual block is computed in the following manner:

    .. math:: output = activation(layers(x) + shortcut(x))

    where

    - ``x``: Input to the Module
    - ``layers``: The feed forward network
    - ``shortcut``: The function to be applied along the skip connection
    - ``activation``: The activation function applied at the end of the residual block

    Args:
        filters (list): A list of the filter sizes. For ex, if the input has a channel
            dimension of 16, and you want 3 convolution layers and the final output to have a
            channel dimension of 16, then the list would be [16, 32, 64, 16].
        kernels (list): A list of the kernel sizes. Each kernel size can be an integer or a
            tuple, similar to Pytorch convention. The length of the ``kernels`` list must be
            1 less than the ``filters`` list.
        strides (list, optional): A list of the strides for each convolution layer.
        paddings (list, optional): A list of the padding in each convolution layer.
        activation (torch.nn.Module, optional): The activation to be used after every convolution
            layer.
        batchnorm (bool, optional): If set to ``False``, batch normalization is not used after
            every convolution layer.
        shortcut (torch.nn.Module, optional): The function to be applied on the input along the
            skip connection.
        last_activation (torch.nn.Module, optional): The activation to be applied at the end of
            the residual block.
    """
    def __init__(self, filters, kernels, strides=None, paddings=None, activation=nn.ReLU,
                 batchnorm=True, shortcut=None, last_activation=None):
        super(ResidualBlock2d, self).__init__()
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
        self.last_activation = last_activation

    def forward(self, x):
        r"""Computes the output of the residual block

        Args:
            x (torch.Tensor): A 4D Torch Tensor which is the input to the Residual Block.

        Returns:
            4D Torch Tensor after applying the desired functions as specified while creating the
            object.
        """
        out = self.layers(x)
        if self.shortcut is not None:
            out += self.shortcut(x)
        if self.last_activation is None:
            return out
        else:
            return self.last_activation(out)
