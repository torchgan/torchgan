import unittest
import torch
from sys import path
path.append('..')
from torchgan.models import *


class TestModels(unittest.TestCase):
    def test_dcgan_generator(self):
        encodings = [50, 100]
        channels = [3, 4]
        step = [64, 128]
        batchnorm = [True, False]
        nonlinearities = [None, torch.nn.ELU(0.5)]
        last_nonlinearity = [None, torch.nn.RReLU(0.25)]
        for i in range(2):
            ch = step[i]
            x = torch.randn(10, encodings[i], 1, 1)
            gen = DCGANGenerator(encodings[i], channels[i], ch, batchnorm[i], nonlinearities[i], last_nonlinearity[i])
            y = gen(x)
            assert y.shape == (10, channels[i], 64, 64)

    def test_dcgan_discriminator(self):
        channels = [3, 4]
        step = [64, 128]
        batchnorm = [True, False]
        nonlinearities = [None, torch.nn.ELU(0.5)]
        last_nonlinearity = [None, torch.nn.RReLU(0.25)]
        for i in range(2):
            x = torch.randn(10, channels[i], 64, 64)
            D = DCGANDiscriminator(channels[i], step[i], batchnorm[i], nonlinearities[i], last_nonlinearity[i])
            y = D(x)
            assert y.shape == (10, 1, 1, 1)
