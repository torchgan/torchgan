import torch
import torch.nn as nn
import torch.nn.functional as F

from .dcgan import DCGANDiscriminator, DCGANGenerator

__all__ = ["ACGANGenerator", "ACGANDiscriminator"]


class ACGANGenerator(DCGANGenerator):
    r"""Auxiliary Classifier GAN (ACGAN) generator based on a DCGAN model from
    `"Conditional Image Synthesis With Auxiliary Classifier GANs
    by Odena et. al. " <https://arxiv.org/abs/1610.09585>`_ paper

    Args:
        num_classes (int): Total classes present in the dataset.
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
    """

    def __init__(
        self,
        num_classes,
        encoding_dims=100,
        out_size=32,
        out_channels=3,
        step_channels=64,
        batchnorm=True,
        nonlinearity=None,
        last_nonlinearity=None,
    ):
        super(ACGANGenerator, self).__init__(
            encoding_dims,
            out_size,
            out_channels,
            step_channels,
            batchnorm,
            nonlinearity,
            last_nonlinearity,
            label_type="generated",
        )
        self.encoding_dims = encoding_dims
        self.num_classes = num_classes
        self.label_embeddings = nn.Embedding(self.num_classes, self.encoding_dims)

    def forward(self, z, y):
        r"""Calculates the output tensor on passing the encoding ``z`` through the Generator.

        Args:
            z (torch.Tensor): A 2D torch tensor of the encoding sampled from a probability
                distribution.
            y (torch.Tensor): The labels corresponding to the encoding ``z``.

        Returns:
            A 4D torch.Tensor of the generated Images conditioned on ``y``.
        """
        y_emb = self.label_embeddings(y.type(torch.LongTensor).to(y.device))
        return super(ACGANGenerator, self).forward(torch.mul(y_emb, z))

    def sampler(self, sample_size, device):
        return [
            torch.randn(sample_size, self.encoding_dims, device=device),
            torch.randint(0, self.num_classes, (sample_size,), device=device),
        ]


class ACGANDiscriminator(DCGANDiscriminator):
    r"""Auxiliary Classifier GAN (ACGAN) discriminator based on a DCGAN model from
    `"Conditional Image Synthesis With Auxiliary Classifier GANs
    by Odena et. al. " <https://arxiv.org/abs/1610.09585>`_ paper

    Args:
        num_classes (int): Total classes present in the dataset.
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
    """

    def __init__(
        self,
        num_classes,
        in_size=32,
        in_channels=3,
        step_channels=64,
        batchnorm=True,
        nonlinearity=None,
        last_nonlinearity=None,
    ):
        super(ACGANDiscriminator, self).__init__(
            in_size,
            in_channels,
            step_channels,
            batchnorm,
            nonlinearity,
            last_nonlinearity,
            label_type="none",
        )
        last_nl = nn.LeakyReLU(0.2) if last_nonlinearity is None else last_nonlinearity
        self.input_dims = in_channels
        self.num_classes = num_classes
        d = self.n * 2 ** (in_size.bit_length() - 4)
        self.aux = nn.Sequential(
            nn.Conv2d(d, self.num_classes, 4, 1, 0, bias=False), last_nl
        )

    def forward(self, x, mode="discriminator", feature_matching=False):
        r"""Calculates the output tensor on passing the image ``x`` through the Discriminator.

        Args:
            x (torch.Tensor): A 4D torch tensor of the image.
            mode (str, optional): Option to choose the mode of the ACGANDiscriminator. Setting it to
                'discriminator' gives the probability of the image being fake/real, 'classifier' allows
                it to make a prediction about the class of the image and anything else leads to
                returning both the values.
            feature_matching (bool, optional): Returns the activation from a predefined intermediate
                layer.

        Returns:
            A 1D torch.Tensor of the probability of each image being real.
        """
        x = self.model(x)
        if feature_matching is True:
            return x
        if mode == "discriminator":
            dx = self.disc(x)
            return dx.view(dx.size(0))
        elif mode == "classifier":
            cx = self.aux(x)
            return cx.view(cx.size(0), cx.size(1))
        else:
            dx = self.disc(x)
            cx = self.aux(x)
            return dx.view(dx.size(0)), cx.view(cx.size(0), cx.size(1))
