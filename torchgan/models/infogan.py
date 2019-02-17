import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F

from .dcgan import DCGANDiscriminator, DCGANGenerator

__all__ = ["InfoGANGenerator", "InfoGANDiscriminator"]


class InfoGANGenerator(DCGANGenerator):
    r"""Generator for InfoGAN based on the Deep Convolutional GAN (DCGAN) architecture, from
    `"InfoGAN : Interpretable Representation Learning With Information Maximizing Generative Aversarial Nets
    by Chen et. al. " <https://arxiv.org/abs/1606.03657>`_ paper

    Args:
        dim_dis (int): Dimension of the discrete latent code sampled from the prior.
        dim_cont (int): Dimension of the continuous latent code sampled from the prior.
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

    Example:
        >>> import torchgan.models as models
        >>> G = models.InfoGANGenerator(10, 30)
        >>> z = torch.randn(10, 100)
        >>> c_cont = torch.randn(10, 10)
        >>> c_dis = torch.randn(10, 30)
        >>> x = G(z, c_cont, c_dis)
    """

    def __init__(
        self,
        dim_dis,
        dim_cont,
        encoding_dims=100,
        out_size=32,
        out_channels=3,
        step_channels=64,
        batchnorm=True,
        nonlinearity=None,
        last_nonlinearity=None,
    ):
        super(InfoGANGenerator, self).__init__(
            encoding_dims + dim_dis + dim_cont,
            out_size,
            out_channels,
            step_channels,
            batchnorm,
            nonlinearity,
            last_nonlinearity,
        )
        self.encoding_dims = encoding_dims
        self.dim_cont = dim_cont
        self.dim_dis = dim_dis

    def forward(self, z, c_dis=None, c_cont=None):
        z_cat = (
            torch.cat([z, c_dis, c_cont], dim=1)
            if c_dis is not None and c_cont is not None
            else z
        )
        return super(InfoGANGenerator, self).forward(z_cat)


class InfoGANDiscriminator(DCGANDiscriminator):
    r"""Discriminator for InfoGAN based on the Deep Convolutional GAN (DCGAN) architecture, from
    `"InfoGAN : Interpretable Representation Learning With Information Maximizing Generative Aversarial Nets
    by Chen et. al. " <https://arxiv.org/abs/1606.03657>`_ paper

    The approximate conditional probability distribution over the latent code Q(c|x) is chosen to be a factored
    Gaussian for the continuous latent code and a Categorical distribution for the discrete latent code

    Args:
        dim_dis (int): Dimension of the discrete latent code sampled from the prior.
        dim_cont (int): Dimension of the continuous latent code sampled from the prior.
        encoding_dims (int, optional): Dimension of the encoding vector sampled from the noise prior.
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
        latent_nonlinearity (torch.nn.Module, optional): Nonlinearity to be used in the ``dist_conv``.
            Defaults to ``LeakyReLU(0.2)`` when None is passed.
    Example:
        >>> import torchgan.models as models
        >>> D = models.InfoGANDiscriminator(10, 30)
        >>> x = torch.randn(10, 3, 32, 32)
        >>> score, q_categorical, q_gaussian = D(x, return_latents=True)
    """

    def __init__(
        self,
        dim_dis,
        dim_cont,
        in_size=32,
        in_channels=3,
        step_channels=64,
        batchnorm=True,
        nonlinearity=None,
        last_nonlinearity=None,
        latent_nonlinearity=None,
    ):
        self.dim_cont = dim_cont
        self.dim_dis = dim_dis
        super(InfoGANDiscriminator, self).__init__(
            in_size,
            in_channels,
            step_channels,
            batchnorm,
            nonlinearity,
            last_nonlinearity,
        )

        self.latent_nl = (
            nn.LeakyReLU(0.2) if latent_nonlinearity is None else latent_nonlinearity
        )
        d = self.n * 2 ** (in_size.bit_length() - 4)
        if batchnorm is True:
            self.dist_conv = nn.Sequential(
                nn.Conv2d(d, d, 4, 1, 0, bias=not batchnorm),
                nn.BatchNorm2d(d),
                self.latent_nl,
            )
        else:
            self.dist_conv = nn.Sequential(
                nn.Conv2d(d, d, 4, 1, 0, bias=not batchnorm), self.latent_nl
            )

        self.dis_categorical = nn.Linear(d, self.dim_dis)

        self.cont_mean = nn.Linear(d, self.dim_cont)
        self.cont_logvar = nn.Linear(d, self.dim_cont)

    def forward(self, x, return_latents=False, feature_matching=False):
        x = self.model(x)
        if feature_matching is True:
            return x
        critic_score = self.disc(x)
        x = self.dist_conv(x).view(-1, x.size(1))
        dist_dis = distributions.OneHotCategorical(logits=self.dis_categorical(x))
        dist_cont = distributions.Normal(
            loc=self.cont_mean(x), scale=torch.exp(0.5 * self.cont_logvar(x))
        )
        return (
            critic_score,
            dist_dis,
            dist_cont if return_latents is True else critic_score,
        )
