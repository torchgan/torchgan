import torch
import torch.nn as nn
import torch.nn.functional as F
from .dcgan import DCGANGenerator, DCGANDiscriminator

__all__ = ['ConditionalGANGenerator', 'ConditionalGANDiscriminator']

class ConditionalGANGenerator(DCGANGenerator):
    r"""Conditional GAN (CGAN) generator based on a DCGAN model from
    `"Conditional Generative Adversarial Nets
    by Mirza et. al. " <https://arxiv.org/abs/1411.1784>`_ paper

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
        super(ConditionalGANGenerator, self).__init__(encoding_dims + num_classes, out_size,
                out_channels, step_channels, batchnorm, nonlinearity, last_nonlinearity, label_type='generated')
        self.encoding_dims = encoding_dims
        self.num_classes = num_classes
        self.label_embeddings = nn.Embedding(self.num_classes, self.num_classes)

    def forward(self, z, y):
        y_emb = self.label_embeddings(y.type(torch.LongTensor).to(y.device))
        return super(ConditionalGANGenerator, self).forward(torch.cat((z, y_emb), dim=1))

    def sampler(self, sample_size, device):
        r"""Function to allow sampling data at inference time.

        Args:
            sample_size (int): The number of images to be generated
            device (torch.device): The device on which the data must be generated

        Returns:
            A list of the items required as input
        """
        return [torch.randn(sample_size, self.encoding_dims, device=device),
                torch.randint(0, self.num_classes, (sample_size,), device=device)]

class ConditionalGANDiscriminator(DCGANDiscriminator):
    r"""Condititional GAN (CGAN) discriminator based on a DCGAN model from
    `"Conditional Generative Adversarial Nets
    by Mirza et. al. " <https://arxiv.org/abs/1411.1784>`_ paper

    Args:
        num_classes (int) : Total classes present in the dataset.
        encoding_dims (int, optional) : Dimension of the encoding vector sampled from the noise prior. Default 100
        out_channels (int, optional) : Number of channels in the output Tensor.
        step_channels (int, optional) : Number of channels in multiples of which the CGAN steps up
                                        the convolutional features
                                        The step up is done as dim `z -> d - > 2 * d -> 4 * d - > 8 * d`
                                        where d = step_channels.
        batchnorm (bool, optional) : If True, use batch normalization in the convolutional layers of the generator.
        nonlinearity (torch.nn.Module, optional) : Nonlinearity to be used in the intermediate convolutional layers
                                                  Defaults to LeakyReLU(0.2) when None is passed.
        last_nonlinearity (torch.nn.Module, optional) : Nonlinearity to be used in the final convolutional layer
                                                       Defaults to tanh when None is passed.

    """
    def __init__(self, num_classes, in_size=32, in_channels=3, step_channels=64, batchnorm=True,
                 nonlinearity=None, last_nonlinearity=None):
        super(ConditionalGANDiscriminator, self).__init__(in_size, in_channels + num_classes, step_channels,
                batchnorm, nonlinearity, last_nonlinearity, label_type='required')
        self.input_dims = in_channels
        self.num_classes = num_classes
        self.label_embeddings = nn.Embedding(self.num_classes, self.num_classes)

    def forward(self, x, y):
        # TODO(Aniket1998): If directly expanding the embeddings gives poor results,
        # try layers of transposed convolution over the embeddings
        y_emb = self.label_embeddings(y.type(torch.LongTensor).to(y.device))
        y_emb = y_emb.unsqueeze(2).unsqueeze(3).expand(-1, y_emb.size(1), x.size(2), x.size(3))
        return super(ConditionalGANDiscriminator, self).forward(torch.cat((x, y_emb), dim=1))
