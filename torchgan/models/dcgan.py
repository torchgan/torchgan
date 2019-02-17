from math import ceil, log2

import torch.nn as nn
import torch.nn.functional as F

from .model import Discriminator, Generator

__all__ = ["DCGANGenerator", "DCGANDiscriminator"]


class DCGANGenerator(Generator):
    r"""Deep Convolutional GAN (DCGAN) generator from
    `"Unsupervised Representation Learning With Deep Convolutional Generative Aversarial Networks
    by Radford et. al. " <https://arxiv.org/abs/1511.06434>`_ paper

    Args:
        encoding_dims (int, optional): Dimension of the encoding vector sampled from the noise prior.
        out_size (int, optional): Height and width of the input image to be generated. Must be at
            least 16 and should be an exact power of 2.
        out_channels (int, optional): Number of channels in the output Tensor.
        step_channels (int, optional): Number of channels in multiples of which the DCGAN steps up
            the convolutional features. The step up is done as dim :math:`z \rightarrow d \rightarrow
            2 \times d \rightarrow 4 \times d \rightarrow 8 \times d` where :math:`d` = step_channels.
        batchnorm (bool, optional): If True, use batch normalization in the convolutional layers of
            the generator.
        nonlinearity (torch.nn.Module, optional): Nonlinearity to be used in the intermediate
            convolutional layers. Defaults to ``LeakyReLU(0.2)`` when None is passed.
        last_nonlinearity (torch.nn.Module, optional): Nonlinearity to be used in the final
            convolutional layer. Defaults to ``Tanh()`` when None is passed.
        label_type (str, optional): The type of labels expected by the Generator. The available
            choices are 'none' if no label is needed, 'required' if the original labels are
            needed and 'generated' if labels are to be sampled from a distribution.
    """

    def __init__(
        self,
        encoding_dims=100,
        out_size=32,
        out_channels=3,
        step_channels=64,
        batchnorm=True,
        nonlinearity=None,
        last_nonlinearity=None,
        label_type="none",
    ):
        super(DCGANGenerator, self).__init__(encoding_dims, label_type)
        if out_size < 16 or ceil(log2(out_size)) != log2(out_size):
            raise Exception(
                "Target Image Size must be at least 16*16 and an exact power of 2"
            )
        num_repeats = out_size.bit_length() - 4
        self.ch = out_channels
        self.n = step_channels
        use_bias = not batchnorm
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        last_nl = nn.Tanh() if last_nonlinearity is None else last_nonlinearity
        model = []
        d = int(self.n * (2 ** num_repeats))
        if batchnorm is True:
            model.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.encoding_dims, d, 4, 1, 0, bias=use_bias),
                    nn.BatchNorm2d(d),
                    nl,
                )
            )
            for i in range(num_repeats):
                model.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(d, d // 2, 4, 2, 1, bias=use_bias),
                        nn.BatchNorm2d(d // 2),
                        nl,
                    )
                )
                d = d // 2
        else:
            model.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.encoding_dims, d, 4, 1, 0, bias=use_bias),
                    nl,
                )
            )
            for i in range(num_repeats):
                model.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(d, d // 2, 4, 2, 1, bias=use_bias), nl
                    )
                )
                d = d // 2

        model.append(
            nn.Sequential(nn.ConvTranspose2d(d, self.ch, 4, 2, 1, bias=True), last_nl)
        )
        self.model = nn.Sequential(*model)
        self._weight_initializer()

    def forward(self, x, feature_matching=False):
        r"""Calculates the output tensor on passing the encoding ``x`` through the Generator.

        Args:
            x (torch.Tensor): A 2D torch tensor of the encoding sampled from a probability
                distribution.
            feature_matching (bool, optional): Returns the activation from a predefined intermediate
                layer.

        Returns:
            A 4D torch.Tensor of the generated image.
        """
        x = x.view(-1, x.size(1), 1, 1)
        return self.model(x)


class DCGANDiscriminator(Discriminator):
    r"""Deep Convolutional GAN (DCGAN) discriminator from
    `"Unsupervised Representation Learning With Deep Convolutional Generative Aversarial Networks
    by Radford et. al. " <https://arxiv.org/abs/1511.06434>`_ paper

    Args:
        in_size (int, optional): Height and width of the input image to be evaluated. Must be at
            least 16 and should be an exact power of 2.
        in_channels (int, optional): Number of channels in the input Tensor.
        step_channels (int, optional): Number of channels in multiples of which the DCGAN steps up
            the convolutional features. The step up is done as dim :math:`z \rightarrow d \rightarrow
            2 \times d \rightarrow 4 \times d \rightarrow 8 \times d` where :math:`d` = step_channels.
        batchnorm (bool, optional): If True, use batch normalization in the convolutional layers of
            the generator.
        nonlinearity (torch.nn.Module, optional): Nonlinearity to be used in the intermediate
            convolutional layers. Defaults to ``LeakyReLU(0.2)`` when None is passed.
        last_nonlinearity (torch.nn.Module, optional): Nonlinearity to be used in the final
            convolutional layer. Defaults to ``Tanh()`` when None is passed.
        label_type (str, optional): The type of labels expected by the Generator. The available
            choices are 'none' if no label is needed, 'required' if the original labels are
            needed and 'generated' if labels are to be sampled from a distribution.
    """

    def __init__(
        self,
        in_size=32,
        in_channels=3,
        step_channels=64,
        batchnorm=True,
        nonlinearity=None,
        last_nonlinearity=None,
        label_type="none",
    ):
        super(DCGANDiscriminator, self).__init__(in_channels, label_type)
        if in_size < 16 or ceil(log2(in_size)) != log2(in_size):
            raise Exception(
                "Input Image Size must be at least 16*16 and an exact power of 2"
            )
        num_repeats = in_size.bit_length() - 4
        self.n = step_channels
        use_bias = not batchnorm
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        last_nl = nn.LeakyReLU(0.2) if last_nonlinearity is None else last_nonlinearity
        d = self.n
        model = [nn.Sequential(nn.Conv2d(self.input_dims, d, 4, 2, 1, bias=True), nl)]
        if batchnorm is True:
            for i in range(num_repeats):
                model.append(
                    nn.Sequential(
                        nn.Conv2d(d, d * 2, 4, 2, 1, bias=use_bias),
                        nn.BatchNorm2d(d * 2),
                        nl,
                    )
                )
                d *= 2
        else:
            for i in range(num_repeats):
                model.append(
                    nn.Sequential(nn.Conv2d(d, d * 2, 4, 2, 1, bias=use_bias), nl)
                )
                d *= 2
        self.disc = nn.Sequential(nn.Conv2d(d, 1, 4, 1, 0, bias=use_bias), last_nl)
        self.model = nn.Sequential(*model)
        self._weight_initializer()

    def forward(self, x, feature_matching=False):
        r"""Calculates the output tensor on passing the image ``x`` through the Discriminator.

        Args:
            x (torch.Tensor): A 4D torch tensor of the image.
            feature_matching (bool, optional): Returns the activation from a predefined intermediate
                layer.

        Returns:
            A 1D torch.Tensor of the probability of each image being real.
        """
        x = self.model(x)
        if feature_matching is True:
            return x
        else:
            x = self.disc(x)
            return x.view(x.size(0))
