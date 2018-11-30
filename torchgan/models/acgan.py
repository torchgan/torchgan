import torch
import torch.nn as nn
import torch.nn.functional as F
from .dcgan import DCGANGenerator, DCGANDiscriminator

__all__ = ['ACGANGenerator', 'ACGANDiscriminator']

class ACGANGenerator(DCGANGenerator):
    r"""Auxiliary Classifier GAN (ACGAN) generator based on a DCGAN model from
    `"Conditional Image Synthesis With Auxiliary Classifier GANs
    by Odena et. al. " <https://arxiv.org/abs/1610.09585>`_ paper

    Args:
        num_classes (int) : Total classes present in the dataset.
        encoding_dims (int, optional) : Dimension of the encoding vector sampled from the noise prior. Default 100
        out_channels (int, optional) : Number of channels in the output Tensor.
        step_channels (int, optional) : Number of channels in multiples of which the CGAN steps up
                                        the convolutional features
                                        The step up is done as dim `z -> d - > 2 * d -> 4 * d - > 8 * d`
                                        where d = step_channels.
        batchnorm (bool, optional) : If True, use batch normalization in the convolutional layers of the generator
        nonlinearity (torch.nn.Module, optional) : Nonlinearity to be used in the intermediate convolutional layers
                                                  Defaults to LeakyReLU(0.2) when None is passed.
        last_nonlinearity (torch.nn.Module, optional) : Nonlinearity to be used in the final convolutional layer
                                                       Defaults to tanh when None is passed.

    """
    def __init__(self, num_classes, encoding_dims=100, out_size=32, out_channels=3,
                 step_channels=64, batchnorm=True, nonlinearity=None, last_nonlinearity=None):
        super(ACGANGenerator, self).__init__(encoding_dims, out_size,
                out_channels, step_channels, batchnorm, nonlinearity, last_nonlinearity, label_type='generated')
        self.encoding_dims = encoding_dims
        self.num_classes = num_classes
        self.label_embeddings = nn.Embedding(self.num_classes, self.encoding_dims)

    def forward(self, z, y):
        y_emb = self.label_embeddings(y.type(torch.LongTensor).to(y.device))
        return super(ACGANGenerator, self).forward(torch.mul(y_emb, z))

class ACGANDiscriminator(DCGANDiscriminator):
    r"""Auxiliary Classifier GAN (ACGAN) discriminator based on a DCGAN model from
    `"Conditional Image Synthesis With Auxiliary Classifier GANs
    by Odena et. al. " <https://arxiv.org/abs/1610.09585>`_ paper

    Args:
        num_classes (int) : Total classes present in the dataset.
        encoding_dims (int, optional) : Dimension of the encoding vector sampled from the noise prior. Default 100
        out_channels (int, optional) : Number of channels in the output Tensor.
        step_channels (int, optional) : Number of channels in multiples of which the CGAN steps up
                                        the convolutional features
                                        The step up is done as dim `z -> d - > 2 * d -> 4 * d - > 8 * d`
                                        where d = step_channels.
        batchnorm (bool, optional) : If True, use batch normalization in the convolutional layers of the generator
        nonlinearity (torch.nn.Module, optional) : Nonlinearity to be used in the intermediate convolutional layers
                                                  Defaults to LeakyReLU(0.2) when None is passed.
        last_nonlinearity (torch.nn.Module, optional) : Nonlinearity to be used in the final convolutional layer
                                                       Defaults to tanh when None is passed.

    """
    def __init__(self, num_classes, in_size=32, in_channels=3, step_channels=64, batchnorm=True,
            nonlinearity=None, last_nonlinearity=None):
        super(ACGANDiscriminator, self).__init__(in_size, in_channels, step_channels,
                batchnorm, nonlinearity, last_nonlinearity, label_type='none')
        last_nl = nn.LeakyReLU(0.2) if last_nonlinearity is None else last_nonlinearity
        self.input_dims = in_channels
        self.num_classes = num_classes
        self.conv = self.model[:len(self.model) - 1]
        self.disc = self.model[len(self.model) - 1]
        d = self.n * 2 ** (in_size.bit_length() - 4)
        self.aux = nn.Sequential(
            nn.Conv2d(d, self.num_classes, 4, 1, 0, bias=False), last_nl)

    def forward(self, x, mode='discriminator'):
        x = self.conv(x)
        if mode == 'discriminator':
            dx = self.disc(x)
            return dx.view(dx.size(0),)
        elif mode == 'classifier':
            cx = self.aux(x)
            return cx.view(cx.size(0), cx.size(1))
        else:
            dx = self.disc(x)
            cx = self.aux(x)
            return dx.view(dx.size(0),), cx.view(cx.size(0), cx.size(1))
