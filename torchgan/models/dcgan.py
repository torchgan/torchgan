import torch.nn as nn
import torch.nn.functional as F
from .model import Generator, Discriminator

__all__ = ['DCGANGenerator', 'DCGANDiscriminator']

class DCGANGenerator(Generator):
    r"""Deep Convolutional GAN (DCGAN) generator from
    "Unsupervised Representation Learning With Deep Convolutional Generative Aversarial Networks
     by Radford et. al. " <https://arxiv.org/abs/1511.06434>

     Args:
        encoding_dims (int, optional) : Dimension of the encoding vector sampled from the noise prior. Default 100
        out_channels (int, optional) : Number of channels in the output Tensor. Default 3
        step_channels (int, optional) : Number of channels in multiples of which the DCGAN steps up
                                        the convolutional features
                                        The step up is done as dim z -> d - > 2 * d -> 4 * d - > 8 * d
                                        where d = step_channels. Default 64

        batchnorm (bool, optional) : If True, use batch normalization in the convolutional layers of the generator
                                     Default True

        nonlinearity(torch.nn.Module, optional) : Nonlinearity to be used in the intermediate convolutional layers
                                                  Defaults to LeakyReLU(0.2) when None is passed. Default None

        last_nonlinearity(torch.nn.Module, optional) : Nonlinearity to be used in the final convolutional layer
                                                       Defaults to tanh when None is passed. Default None

    """
    def __init__(self, encoding_dims=100, out_channels=3, step_channels=64,
                 batchnorm=True, nonlinearity=None, last_nonlinearity=None):
        super(DCGANGenerator, self).__init__(self, encoding_dims)
        self.ch = out_channels
        self.step_ch = step_channels
        self.batchnorm = batchnorm

        self.nl = getattr(F, nonlinearity)
        self.last_nl = getattr(F, last_nonlinearity)

        if batchnorm is True:
            use_bias = False
        else:
            use_bias = True

        if nonlinearity is None:
            self.nl = nn.LeakyReLU(0.2)
        else:
            self.nl = nonlinearity

        if nonlinearity is None:
            self.last_nl = nn.Tanh()
        else:
            self.last_nl = last_nonlinearity

        self.conv1 = nn.ConvTranspose2d(self.encoding_dims, self.step_ch * 8, 4, 1, 0, bias=use_bias)
        self.conv2 = nn.ConvTranspose2d(self.step_ch * 8, self.step_ch * 4, 4, 2, 1, bias=use_bias)
        self.conv3 = nn.ConvTranspose2d(self.step_ch * 4, self.step_ch * 2, 4, 2, 1, bias=use_bias)
        self.conv4 = nn.ConvTranspose2d(self.step_ch * 2, self.step_ch, 4, 2, 1, bias=use_bias)
        self.conv5 = nn.ConvTranspose2d(self.step_ch, self.ch, 4, 2, 1, bias=use_bias)

        if batchnorm is True:
            self.bn1 = nn.BatchNorm2d(self.step_ch * 8)
            self.bn2 = nn.BatchNorm2d(self.step_ch * 4)
            self.bn3 = nn.BatchNorm2d(self.step_ch * 2)
            self.bn4 = nn.BatchNorm2d(self.step_ch)

        def forward(self, x):
            if self.batchnorm is True:
                self.conv1_feat = self.nl(self.bn1(self.conv1(x)))
                self.conv2_feat = self.nl(self.bn2(self.conv2(self.conv1_feat)))
                self.conv3_feat = self.nl(self.bn3(self.conv3(self.conv2_feat)))
                self.conv4_feat = self.nl(self.bn4(self.conv4(self.conv3_feat)))
            else:
                self.conv1_feat = self.nl(self.conv1(x))
                self.conv2_feat = self.nl(self.conv2(self.conv1_feat))
                self.conv3_feat = self.nl(self.conv3(self.conv3_feat))
                self.conv4_feat = self.nl(self.conv4(self.conv4_feat))

            self.conv5_feat = self.last_nl(self.conv5(self.conv4_feat))
            return self.conv5_feat


class DCGANDiscriminator(Generator):
    r"""Deep Convolutional GAN (DCGAN) discriminator from
    "Unsupervised Representation Learning With Deep Convolutional Generative Aversarial Networks
     by Radford et. al. " <https://arxiv.org/abs/1511.06434>

     Args:
        encoding_dims (int, optional) : Dimension of the encoding vector sampled from the noise prior. Default 100
        out_channels (int, optional) : Number of channels in the output Tensor. Default 3
        step_channels (int, optional) : Number of channels in multiples of which the DCGAN steps up
                                        the convolutional features
                                        The step up is done as dim z -> d - > 2 * d -> 4 * d - > 8 * d
                                        where d = step_channels. Default 64

        batchnorm (bool, optional) : If True, use batch normalization in the convolutional layers of the generator
                                     Default True

        nonlinearity(torch.nn.Module, optional) : Nonlinearity to be used in the intermediate convolutional layers
                                                  Defaults to LeakyReLU(0.2) when None is passed. Default None

        last_nonlinearity(toch.nn.Module, optional) : Nonlinearity to be used in the final convolutional layer
                                                      Defaults to sigmoid when None is passed. Default None

    """

    def __init__(self, in_channels=3, step_channels=64, batchnorm=True,
                 nonlinearity=None, last_nonlinearity=None):
        super(DCGANDiscriminator, self).__init__(in_channels)
        self.step_ch = step_channels
        self.batchnorm = batchnorm

        self.nl = getattr(F, nonlinearity)
        self.last_nl = getattr(F, last_nonlinearity)

        if batchnorm is True:
            use_bias = False
        else:
            use_bias = True

        if nonlinearity is None:
            self.nl = nn.LeakyReLU(0.2)
        else:
            self.nl = nonlinearity

        if nonlinearity is None:
            self.last_nl = nn.Sigmoid()
        else:
            self.last_nl = last_nonlinearity

        self.conv1 = nn.Conv2d(self.input_dims, self.step_ch, 4, 2, 1, bias=use_bias)
        self.conv2 = nn.Conv2d(self.step_ch, self.step_ch * 2, 4, 2, 1, bias=use_bias)
        self.conv3 = nn.Conv2d(self.step_ch * 2, self.step_ch * 4, 4, 2, 1, bias=use_bias)
        self.conv4 = nn.Conv2d(self.step_ch * 4, self.step_ch * 8, 4, 2, 1, bias=use_bias)
        self.conv5 = nn.Conv2d(self.step_ch * 8, 1, 4, 1, 0, bias=use_bias)

        if batchnorm is True:
            self.bn1 = nn.BatchNorm2d(self.step_ch)
            self.bn2 = nn.BatchNorm2d(self.step_ch * 2)
            self.bn3 = nn.BatchNorm2d(self.step_ch * 4)
            self.bn4 = nn.BatchNorm2d(self.step_ch * 8)

        def forward(self, x):
            if self.batchnorm is True:
                self.conv1_feat = self.nl(self.bn1(self.conv1(x)))
                self.conv2_feat = self.nl(self.bn2(self.conv2(self.conv1_feat)))
                self.conv3_feat = self.nl(self.bn3(self.conv3(self.conv2_feat)))
                self.conv4_feat = self.nl(self.bn4(self.conv4(self.conv3_feat)))
            else:
                self.conv1_feat = self.nl(self.conv1(x))
                self.conv2_feat = self.nl(self.conv2(self.conv1_feat))
                self.conv3_feat = self.nl(self.conv3(self.conv3_feat))
                self.conv4_feat = self.nl(self.conv4(self.conv4_feat))

            self.conv5_feat = self.last_nl(self.conv5(self.conv4_feat))
            return self.conv5_feat
