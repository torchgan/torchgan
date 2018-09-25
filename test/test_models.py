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
        use_cuda = [False, True]
        last_nonlineariy = [None, torch.nn.RReLU(0.25)]
        device = torch.device('cuda:0')
        for i in range(2):
            ch = step[i]
            x = torch.randn(10, encodings[i], 1, 1)
            gen = DCGANGenerator(encodings[i], channels[i], ch, batchnorm[i], nonlinearities[i], last_nonlineariy[i])
            if use_cuda[i] is True:
                x = x.to(device)
                gen.to(device)
            y = gen(x)
            assert y.shape == (10, channels[i], 64, 64)
            assert gen.conv1_feat.shape == (10, ch * 8, 4, 4)
            assert gen.conv2_feat.shape == (10, ch * 4, 8, 8)
            assert gen.conv3_feat.shape == (10, ch * 2, 16, 16)
            assert gen.conv4_feat.shape == (10, ch, 32, 32)

    def test_dcgan_discriminator(self):
        channels = [3, 4]
        step = [64, 128]
        batchnorm = [True, False]
        use_cuda = [False, True]
        nonlinearities = [None, torch.nn.ELU(0.5)]
        last_nonlineariy = [None, torch.nn.RReLU(0.25)]
        device = torch.device('cuda:0')
        for i in range(2):
            ch = step[i]
            x = torch.randn(10, channels[i], 64, 64)
            D = DCGANDiscriminator(channels[i], step[i], batchnorm[i], nonlinearities[i], last_nonlineariy[i])
            if use_cuda[i] is True:
                x = x.to(device)
                D.to(device)
            y = D(x)
            assert y.shape == (10, 1, 1, 1)
            assert D.conv1_feat.shape == (10, ch, 32, 32)
            assert D.conv2_feat.shape == (10, ch * 2, 16, 16)
            assert D.conv3_feat.shape == (10, ch * 4, 8, 8)
            assert D.conv4_feat.shape == (10, ch * 8, 4, 4) 
