import torch
from math import ceil, log
import torch.nn as nn
import torch.nn.functional as F
from ..models import Generator, Discriminator

__all__ = ['AutoEncodingGenerator', 'AutoEncodingDiscriminator']

class AutoEncodingGenerator(Generator):
    r"""Autoencoding Generator for Boundary Equilibrium GAN (BEGAN) from
    `"BEGAN : Boundary Equilibrium Generative Adversarial Networks
    by Berthelot et. al." <https://arxiv.org/abs/1703.10717>`_ paper

    Args:
        encoding_dims (int, optional) : Dimension of the encoding vector sampled from the noise prior. Default 100
        out_size(int, optional)      : Height and Width of the output image to be generated. Must be a power of 2.
                                       Default 32
        out_channels (int, optional) : Number of channels in the output Tensor.
        step_channels (int, optional) : Number of channels in multiples of which the DCGAN steps up
                                        the convolutional features
                                        The step up is done as dim `z -> d - > 2 * d -> 4 * d - > 8 * d`
                                        where d = step_channels.
        batchnorm (bool, optional) : If True, use batch normalization in the convolutional layers of the generator.
        nonlinearity (torch.nn.Module, optional) : Nonlinearity to be used in the intermediate convolutional layers
                                                  Defaults to LeakyReLU(0.2) when None is passed.
        last_nonlinearity (torch.nn.Module, optional) : Nonlinearity to be used in the final convolutional layer
                                                       Defaults to tanh when None is passed.
    """
    def __init__(self, encoding_dims=100, out_size=32, out_channels=3, step_channels=64, scale_factor=2,
                 batchnorm=True, nonlinearity=None, last_nonlinearity=None, label_type='none'):
        super(AutoEncodingGenerator, self).__init__(encoding_dims, label_type)
        if out_size < (scale_factor ** 4) or ceil(log(out_size, scale_factor)) != log(out_size, scale_factor):
            raise Exception('Target image size must be at least {} and a perfect power of {}'.format(scale_factor ** 4,
                scale_factor))
        num_repeats = int(log(out_size, scale_factor)) - 3
        same_filters = scale_factor + 1
        same_pad = scale_factor // 2
        if scale_factor == 2:
            upsample_filters = 3
            upsample_stride = 2
            upsample_pad = 1
            upsample_output_pad = 1
        else:
            upsample_filters = scale_factor
            upsample_stride = scale_factor
            upsample_pad = 0
            upsample_output_pad = 0
        self.ch = out_channels
        self.n = step_channels
        use_bias = not batchnorm
        nl = nn.ELU() if nonlinearity is None else nonlinearity
        last_nl = nn.Tanh() if last_nonlinearity is None else last_nonlinearity
        init_dim = scale_factor ** 3
        self.init_dim = init_dim

        if batchnorm is True:
            self.fc = nn.Sequential(
                nn.Linear(self.encoding_dims, (init_dim ** 2) * self.n),
                nn.BatchNorm1d((init_dim ** 2) * self.n), nl)
            initial_unit = nn.Sequential(
                nn.Conv2d(self.n, self.n, same_filters, 1, same_pad, bias=use_bias),
                nn.BatchNorm2d(self.n), nl,
                nn.Conv2d(self.n, self.n, same_filters, 1, same_pad, bias=use_bias),
                nn.BatchNorm2d(self.n), nl)
            upsample_unit = nn.Sequential(
                nn.ConvTranspose2d(self.n, self.n, upsample_filters,
                    upsample_stride, upsample_pad, upsample_output_pad, bias=use_bias),
                nn.BatchNorm2d(self.n), nl,
                nn.Conv2d(self.n, self.n, same_filters, 1, same_pad, bias=use_bias),
                nn.BatchNorm2d(self.n), nl)
        else:
            self.fc = nn.Sequential(
                nn.Linear(self.encoding_dims, (init_dim ** 2) * self.n), nl)
            initial_unit = nn.Sequential(
                nn.Conv2d(self.n, self.n, same_filters, 1, same_pad, bias=use_bias), nl,
                nn.Conv2d(self.n, self.n, same_filters, 1, same_pad, bias=use_bias), nl)
            upsample_unit = nn.Sequential(
                nn.ConvTranspose2d(self.n, self.n, upsample_filters,
                    upsample_stride, upsample_pad, upsample_output_pad, bias=use_bias), nl,
                nn.Conv2d(self.n, self.n, same_filters, 1, same_pad, bias=use_bias), nl)

        last_unit = nn.Sequential(
            nn.Conv2d(self.n, self.ch, same_filters, 1, same_pad, bias=True), last_nl)
        model = [initial_unit]
        for i in range(num_repeats):
            model.append(upsample_unit)
            out_size = out_size // scale_factor
        model.append(last_unit)
        self.model = nn.Sequential(*model)
        self._weight_initializer()

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.n, self.init_dim, self.init_dim)
        return self.model(x)


class AutoEncodingDiscriminator(Discriminator):
    r"""Autoencoding Generator for Boundary Equilibrium GAN (BEGAN) from
    `"BEGAN : Boundary Equilibrium Generative Adversarial Networks
    by Berthelot et. al." <https://arxiv.org/abs/1703.10717>`_ paper

    Args:
        in_size (int, optional)     : Height and Width of the input image. Must be a power of 2. Default 32
        in_channels (int, optional) : Number of channels in the input Tensor.
        encoding_dims (int, optional) : Dimension of the encoding vector sampled from the noise prior.
        step_channels (int, optional) : Number of channels in multiples of which the DCGAN steps up
                                        the convolutional features
                                        The step up is done as dim `z -> d - > 2 * d -> 4 * d - > 8 * d`
                                        where d = step_channels.
        batchnorm (bool, optional) : If True, use batch normalization in the convolutional layers of the generator.
        nonlinearity (torch.nn.Module, optional) : Nonlinearity to be used in the intermediate convolutional layers
                                                  Defaults to LeakyReLU(0.2) when None is passed.
        last_nonlinearity (toch.nn.Module, optional) : Nonlinearity to be used in the final convolutional layer
                                                      Defaults to sigmoid when None is passed.
    """
    def __init__(self, in_size=32, in_channels=3, encoding_dims=100, step_channels=64, scale_factor=2,
                 batchnorm=True, nonlinearity=None, last_nonlinearity=None, label_type='none'):
        super(AutoEncodingDiscriminator, self).__init__(in_channels, label_type)
        if in_size < (scale_factor ** 4) or ceil(log(in_size, scale_factor)) != log(in_size, scale_factor):
            raise Exception('Input image size must be at least {} and a perfect power of {}'.format(scale_factor ** 4,
                scale_factor))
        num_repeats = int(log(in_size, scale_factor)) - 3
        same_filters = scale_factor + 1
        same_pad = scale_factor // 2
        if scale_factor == 2:
            downsample_filters = 3
            downsample_stride = 2
            downsample_pad = 1
        else:
            downsample_filters = scale_factor
            downsample_stride = scale_factor
            downsample_pad = 0
        self.n = step_channels
        nl = nn.ELU() if nonlinearity is None else nonlinearity
        last_nl = nn.ELU() if last_nonlinearity is None else last_nonlinearity
        use_bias = not batchnorm
        init_dim = scale_factor ** 3
        self.init_dim = init_dim
        model = []
        model.append(nn.Sequential(
            nn.Conv2d(self.input_dims, self.n, same_filters, 1, same_pad, bias=True), nl))
        if batchnorm is True:
            for i in range(1, num_repeats + 1):
                model.append(nn.Sequential(
                    nn.Conv2d(self.n * i, self.n * i, same_filters, 1, same_pad, bias=use_bias),
                    nn.BatchNorm2d(self.n * i), nl,
                    nn.Conv2d(self.n * i, self.n * (i + 1), downsample_filters, downsample_stride,
                        downsample_pad, bias=use_bias),
                    nn.BatchNorm2d(self.n * (i + 1)), nl))
            model.append(nn.Sequential(
                nn.Conv2d(self.n * (num_repeats + 1), self.n * (num_repeats + 1),
                    same_filters, 1, same_pad, bias=use_bias),
                nn.BatchNorm2d(self.n * (num_repeats + 1)), nl,
                nn.Conv2d(self.n * (num_repeats + 1), self.n * (num_repeats + 1),
                    same_filters, 1, same_pad, bias=use_bias),
                nn.BatchNorm2d(self.n * (num_repeats + 1)), nl))
            self.fc = nn.Sequential(
                nn.Linear((init_dim ** 2) * (num_repeats + 1) * self.n, encoding_dims),
                nn.BatchNorm1d(encoding_dims), last_nl)
        else:
            for i in range(1, num_repeats + 1):
                model.append(nn.Sequential(
                    nn.Conv2d(self.n * i, self.n * i, 3, 1, 1, bias=use_bias), nl,
                    nn.Conv2d(self.n * i, self.n * (i + 1), downsample_filters, downsample_stride,
                        downsample_pad, bias=use_bias), nl))
            model.append(nn.Sequential(
                nn.Conv2d(self.n * (num_repeats + 1), self.n * (num_repeats + 1), 3, 1, 1, bias=use_bias), nl,
                nn.Conv2d(self.n * (num_repeats + 1), self.n * (num_repeats + 1), 3, 1, 1, bias=use_bias), nl))
            self.fc = nn.Sequential(
                nn.Linear((init_dim ** 2) * (num_repeats + 1) * self.n, encoding_dims), last_nl)
        self.encoder = nn.Sequential(*model)
        self.decoder = AutoEncodingGenerator(encoding_dims, in_size, in_channels, step_channels, scale_factor,
                                             batchnorm, nonlinearity, last_nonlinearity)
        self._weight_initializer()

    def forward(self, x):
        x1 = self.encoder(x)
        x1 = x1.view(-1, (self.init_dim ** 2) * x1.size(1))
        x1 = self.fc(x1)
        x1 = self.decoder(x1)
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        x1 = x1.view(-1, x1.size(1) * x1.size(2) * x1.size(3))
        return torch.mean((x - x1) ** 2, 1)
