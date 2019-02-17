import torch
import torch.nn as nn

__all__ = [
    "BasicBlock2d",
    "BottleneckBlock2d",
    "TransitionBlock2d",
    "TransitionBlockTranspose2d",
    "DenseBlock2d",
]


class BasicBlock2d(nn.Module):
    r"""Basic Block Module as described in `"Densely Connected Convolutional Networks by Huang et.
    al." <https://arxiv.org/abs/1608.06993>`_

    The output is computed by ``concatenating`` the ``input`` tensor to the ``output`` tensor (of the
    internal model) along the ``channel`` dimension.

    The internal model is simply a sequence of a ``Conv2d`` layer and a ``BatchNorm2d`` layer, if
    activated.

    Args:
        in_channels (int): The channel dimension of the input tensor.
        out_channels (int): The channel dimension of the output tensor.
        kernel (int, tuple): Size of the Convolutional Kernel.
        stride (int, tuple, optional): Stride of the Convolutional Kernel.
        padding (int, tuple, optional): Padding to be applied on the input tensor.
        batchnorm (bool, optional): If ``True``, batch normalization shall be performed.
        nonlinearity (torch.nn.Module, optional): Activation to be applied. Defaults to
                                                  ``torch.nn.LeakyReLU``.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        stride=1,
        padding=0,
        batchnorm=True,
        nonlinearity=None,
    ):
        super(BasicBlock2d, self).__init__()
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        if batchnorm is True:
            self.model = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nl,
                nn.Conv2d(
                    in_channels, out_channels, kernel, stride, padding, bias=False
                ),
            )
        else:
            self.model = nn.Sequential(
                nl,
                nn.Conv2d(
                    in_channels, out_channels, kernel, stride, padding, bias=True
                ),
            )

    def forward(self, x):
        r"""Computes the output of the basic dense block

        Args:
            x (torch.Tensor): The input tensor having channel dimension same as ``in_channels``.

        Returns:
            4D Tensor by concatenating the input to the output of the internal model.
        """
        return torch.cat([x, self.model(x)], 1)


class BottleneckBlock2d(nn.Module):
    r"""Bottleneck Block Module as described in `"Densely Connected Convolutional Networks by Huang
    et. al." <https://arxiv.org/abs/1608.06993>`_

    The output is computed by ``concatenating`` the ``input`` tensor to the ``output`` tensor (of the
    internal model) along the ``channel`` dimension.

    The internal model is simply a sequence of 2 ``Conv2d`` layers and 2 ``BatchNorm2d`` layers, if
    activated. This Module is much more computationally efficient than the ``BasicBlock2d``, and hence
    is more recommended.

    Args:
        in_channels (int): The channel dimension of the input tensor.
        out_channels (int): The channel dimension of the output tensor.
        kernel (int, tuple): Size of the Convolutional Kernel.
        stride (int, tuple, optional): Stride of the Convolutional Kernel.
        padding (int, tuple, optional): Padding to be applied on the input tensor.
        bottleneck_channels (int, optional): The channels in the intermediate convolutional
                                             layer. A higher value will make learning of
                                             more complex functions possible. Defaults to
                                             ``4 * in_channels``.
        batchnorm (bool, optional): If ``True``, batch normalization shall be performed.
        nonlinearity (torch.nn.Module, optional): Activation to be applied. Defaults to
                                                  ``torch.nn.LeakyReLU``.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        stride=1,
        padding=0,
        bottleneck_channels=None,
        batchnorm=True,
        nonlinearity=None,
    ):
        super(BottleneckBlock2d, self).__init__()
        bottleneck_channels = (
            4 * in_channels if bottleneck_channels is None else bottleneck_channels
        )
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        if batchnorm is True:
            self.model = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nl,
                nn.Conv2d(in_channels, bottleneck_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(bottleneck_channels),
                nl,
                nn.Conv2d(
                    bottleneck_channels,
                    out_channels,
                    kernel,
                    stride,
                    padding,
                    bias=False,
                ),
            )
        else:
            self.model = nn.Sequential(
                nl,
                nn.Conv2d(in_channels, bottleneck_channels, 1, 1, 0, bias=True),
                nl,
                nn.Conv2d(
                    bottleneck_channels,
                    out_channels,
                    kernel,
                    stride,
                    padding,
                    bias=True,
                ),
            )

    def forward(self, x):
        r"""Computes the output of the bottleneck dense block

        Args:
            x (torch.Tensor): The input tensor having channel dimension same as ``in_channels``.

        Returns:
            4D Tensor by concatenating the input to the output of the internal model.
        """
        return torch.cat([x, self.model(x)], 1)


class TransitionBlock2d(nn.Module):
    r"""Transition Block Module as described in `"Densely Connected Convolutional Networks by Huang
    et. al." <https://arxiv.org/abs/1608.06993>`_

    This is a simple ``Sequential`` model of a ``Conv2d`` layer and a ``BatchNorm2d`` layer, if
    activated.

    Args:
        in_channels (int): The channel dimension of the input tensor.
        out_channels (int): The channel dimension of the output tensor.
        kernel (int, tuple): Size of the Convolutional Kernel.
        stride (int, tuple, optional): Stride of the Convolutional Kernel.
        padding (int, tuple, optional): Padding to be applied on the input tensor.
        batchnorm (bool, optional): If ``True``, batch normalization shall be performed.
        nonlinearity (torch.nn.Module, optional): Activation to be applied. Defaults to
                                                  ``torch.nn.LeakyReLU``.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        stride=1,
        padding=0,
        batchnorm=True,
        nonlinearity=None,
    ):
        super(TransitionBlock2d, self).__init__()
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        if batchnorm is True:
            self.model = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nl,
                nn.Conv2d(
                    in_channels, out_channels, kernel, stride, padding, bias=False
                ),
            )
        else:
            self.model = nn.Sequential(
                nl,
                nn.Conv2d(
                    in_channels, out_channels, kernel, stride, padding, bias=True
                ),
            )

    def forward(self, x):
        r"""Computes the output of the transition block

        Args:
            x (torch.Tensor): The input tensor having channel dimension same as ``in_channels``.

        Returns:
            4D Tensor by applying the ``model`` on ``x``.
        """
        return self.model(x)


class TransitionBlockTranspose2d(nn.Module):
    r"""Transition Block Transpose Module is constructed by simply reversing the effect of
    Transition Block Module. We replace the ``Conv2d`` layers by ``ConvTranspose2d`` layers.

    Args:
        in_channels (int): The channel dimension of the input tensor.
        out_channels (int): The channel dimension of the output tensor.
        kernel (int, tuple): Size of the Convolutional Kernel.
        stride (int, tuple, optional): Stride of the Convolutional Kernel.
        padding (int, tuple, optional): Padding to be applied on the input tensor.
        batchnorm (bool, optional): If ``True``, batch normalization shall be performed.
        nonlinearity (torch.nn.Module, optional): Activation to be applied. Defaults to
                                                  ``torch.nn.LeakyReLU``.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        stride=1,
        padding=0,
        batchnorm=True,
        nonlinearity=None,
    ):
        super(TransitionBlockTranspose2d, self).__init__()
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        if batchnorm is True:
            self.model = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nl,
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel, stride, padding, bias=False
                ),
            )
        else:
            self.model = nn.Sequential(
                nl,
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel, stride, padding, bias=True
                ),
            )

    def forward(self, x):
        r"""Computes the output of the transition block transpose

        Args:
            x (torch.Tensor): The input tensor having channel dimension same as ``in_channels``.

        Returns:
            4D Tensor by applying the ``model`` on ``x``.
        """
        return self.model(x)


class DenseBlock2d(nn.Module):
    r"""Dense Block Module as described in `"Densely Connected Convolutional Networks by Huang
    et. al." <https://arxiv.org/abs/1608.06993>`_

    Args:
        depth (int): The total number of ``blocks`` that will be present.
        in_channels (int): The channel dimension of the input tensor.
        growth_rate (int): The rate at which the channel dimension increases. The output of
                           the module has a channel dimension of size ``in_channels +
                           depth * growth_rate``.
        block (torch.nn.Module): Should be once of the Densenet Blocks. Forms the building block
                                 for the Dense Block.
        kernel (int, tuple): Size of the Convolutional Kernel.
        stride (int, tuple, optional): Stride of the Convolutional Kernel.
        padding (int, tuple, optional): Padding to be applied on the input tensor.
        batchnorm (bool, optional): If ``True``, batch normalization shall be performed.
        nonlinearity (torch.nn.Module, optional): Activation to be applied. Defaults to
                                                  ``torch.nn.LeakyReLU``.
    """

    def __init__(
        self,
        depth,
        in_channels,
        growth_rate,
        block,
        kernel,
        stride=1,
        padding=0,
        batchnorm=True,
        nonlinearity=None,
    ):
        super(DenseBlock2d, self).__init__()
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        model = []
        for i in range(depth):
            # FIXME(Aniket1998): There is no way to pass an option for bottleneck channels
            model.append(
                block(
                    in_channels + i * growth_rate,
                    growth_rate,
                    kernel,
                    stride,
                    padding,
                    batchnorm=batchnorm,
                    nonlinearity=nl,
                )
            )
        self.model = nn.Sequential(*model)

    def forward(self, x):
        r"""Computes the output of the transition block transpose

        Args:
            x (torch.Tensor): The input tensor having channel dimension same as ``in_channels``.

        Returns:
            4D Tensor by applying the ``model`` on ``x``.
        """
        return self.model(x)
