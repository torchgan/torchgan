import os
import sys
import unittest

import torch
import torch.distributions as distributions
from torchgan.models import *

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


class TestModels(unittest.TestCase):
    def test_dcgan_generator(self):
        encodings = [50, 100]
        channels = [3, 4]
        out_size = [32, 256]
        step = [64, 128]
        batchnorm = [True, False]
        nonlinearities = [None, torch.nn.ReLU()]
        last_nonlinearity = [None, torch.nn.LeakyReLU()]
        for i in range(2):
            gen = DCGANGenerator(
                encodings[i],
                out_size[i],
                channels[i],
                step[i],
                batchnorm[i],
                nonlinearities[i],
                last_nonlinearity[i],
            )
            z = torch.rand(10, encodings[i])
            x = gen(z)
            assert x.shape == (10, channels[i], out_size[i], out_size[i])

    def test_dcgan_discriminator(self):
        channels = [3, 4]
        step = [64, 128]
        in_size = [32, 64]
        batchnorm = [True, False]
        nonlinearities = [None, torch.nn.ReLU()]
        last_nonlinearity = [None, torch.nn.LeakyReLU()]
        for i in range(2):
            dis = DCGANDiscriminator(
                in_size[i],
                channels[i],
                step[i],
                batchnorm[i],
                nonlinearities[i],
                last_nonlinearity[i],
            )
            x = torch.rand(10, channels[i], in_size[i], in_size[i])
            loss = dis(x)
            assert loss.shape == (10,)

    def test_conditional_gan_generator(self):
        encodings = [50, 100]
        channels = [3, 4]
        out_size = [32, 64]
        classes = [5, 10]
        step = [64, 128]
        batchnorm = [True, False]
        nonlinearities = [None, torch.nn.ELU(0.5)]
        last_nonlinearity = [None, torch.nn.RReLU(0.25)]
        for i in range(2):
            ch = step[i]
            x = torch.randn(10, encodings[i])
            gen = ConditionalGANGenerator(
                classes[i],
                encodings[i],
                out_size[i],
                channels[i],
                ch,
                batchnorm[i],
                nonlinearities[i],
                last_nonlinearity[i],
            )
            labels = torch.randint(0, classes[i], (10,))
            y = gen(x, labels)
            assert y.shape == (10, channels[i], out_size[i], out_size[i])

    def test_conditional_gan_discriminator(self):
        channels = [3, 4]
        in_size = [32, 64]
        classes = [5, 10]
        step = [64, 128]
        batchnorm = [True, False]
        nonlinearities = [None, torch.nn.ELU(0.5)]
        last_nonlinearity = [None, torch.nn.RReLU(0.25)]
        for i in range(2):
            ch = step[i]
            x = torch.randn(10, channels[i], in_size[i], in_size[i])
            gen = ConditionalGANDiscriminator(
                classes[i],
                in_size[i],
                channels[i],
                ch,
                batchnorm[i],
                nonlinearities[i],
                last_nonlinearity[i],
            )
            labels = torch.randint(0, classes[i], (10,))
            y = gen(x, labels)
            assert y.shape == (10,)

    def test_acgan_generator(self):
        encodings = [50, 100]
        channels = [3, 4]
        out_size = [32, 64]
        classes = [5, 10]
        step = [64, 128]
        batchnorm = [True, False]
        nonlinearities = [None, torch.nn.ELU(0.5)]
        last_nonlinearity = [None, torch.nn.RReLU(0.25)]
        for i in range(2):
            ch = step[i]
            x = torch.randn(10, encodings[i])
            gen = ACGANGenerator(
                classes[i],
                encodings[i],
                out_size[i],
                channels[i],
                ch,
                batchnorm[i],
                nonlinearities[i],
                last_nonlinearity[i],
            )
            labels = torch.randint(0, classes[i], (10,))
            y = gen(x, labels)
            assert y.shape == (10, channels[i], out_size[i], out_size[i])

    def test_acgan_discriminator(self):
        channels = [3, 4]
        in_size = [32, 64]
        classes = [5, 10]
        step = [64, 128]
        batchnorm = [True, False]
        nonlinearities = [None, torch.nn.ELU(0.5)]
        last_nonlinearity = [None, torch.nn.RReLU(0.25)]
        for i in range(2):
            ch = step[i]
            x = torch.randn(10, channels[i], in_size[i], in_size[i])
            gen = ACGANDiscriminator(
                classes[i],
                in_size[i],
                channels[i],
                ch,
                batchnorm[i],
                nonlinearities[i],
                last_nonlinearity[i],
            )
            dx, cx = gen(x, mode="combine")
            assert dx.shape == (10,)
            assert cx.shape == (10, classes[i])

    def test_infogan_generator(self):
        encodings = [50, 100]
        channels = [3, 4]
        out_size = [32, 64]
        step = [64, 128]
        dim_cont = [10, 20]
        dim_dis = [30, 40]
        batchnorm = [True, False]
        nonlinearities = [None, torch.nn.ELU(0.5)]
        last_nonlinearity = [None, torch.nn.RReLU(0.25)]
        for i in range(2):
            ch = step[i]
            x = torch.randn(10, encodings[i])
            cont = torch.rand(10, dim_cont[i])
            dis = torch.zeros(10, dim_dis[i])
            gen = InfoGANGenerator(
                dim_cont[i],
                dim_dis[i],
                encodings[i],
                out_size[i],
                channels[i],
                ch,
                batchnorm[i],
                nonlinearities[i],
                last_nonlinearity[i],
            )
            y = gen(x, cont, dis)
            assert y.shape == (10, channels[i], out_size[i], out_size[i])

    def test_infogan_discriminator(self):
        channels = [3, 4]
        in_size = [32, 64]
        dim_cont = [10, 20]
        dim_dis = [30, 40]
        step = [64, 128]
        batchnorm = [True, False]
        nonlinearities = [None, torch.nn.ELU(0.5)]
        last_nonlinearity = [None, torch.nn.RReLU(0.25)]
        for i in range(2):
            x = torch.randn(10, channels[i], in_size[i], in_size[i])
            D = InfoGANDiscriminator(
                dim_dis[i],
                dim_cont[i],
                in_size[i],
                channels[i],
                step[i],
                batchnorm[i],
                nonlinearities[i],
                last_nonlinearity[i],
            )
            y, dist_dis, dist_cont = D(x, True)
            assert y.shape == (10, 1, 1, 1)
            assert isinstance(dist_dis, distributions.OneHotCategorical)
            assert isinstance(dist_cont, distributions.Normal)
            assert dist_dis.sample().shape == (10, dim_dis[i])
            assert dist_cont.sample().shape == (10, dim_cont[i])

    def test_autoencoding_generator(self):
        encodings = [50, 100]
        channels = [3, 4]
        out_size = [32, 64]
        step = [64, 128]
        scale = [2, 2]
        batchnorm = [True, False]
        nonlinearities = [None, torch.nn.ReLU()]
        last_nonlinearity = [None, torch.nn.LeakyReLU()]
        for i in range(2):
            gen = AutoEncodingGenerator(
                encodings[i],
                out_size[i],
                channels[i],
                step[i],
                scale[i],
                batchnorm[i],
                nonlinearities[i],
                last_nonlinearity[i],
            )
            z = torch.rand(10, encodings[i])
            x = gen(z)
            assert x.shape == (10, channels[i], out_size[i], out_size[i])

    def test_autoencoding_discriminator(self):
        channels = [3, 4]
        encodings = [50, 100]
        in_size = [32, 64]
        step = [64, 128]
        scale = [2, 2]
        batchnorm = [True, False]
        nonlinearities = [None, torch.nn.ReLU()]
        last_nonlinearity = [None, torch.nn.LeakyReLU()]
        for i in range(2):
            dis = AutoEncodingDiscriminator(
                in_size[i],
                channels[i],
                encodings[i],
                step[i],
                scale[i],
                batchnorm[i],
                nonlinearities[i],
                last_nonlinearity[i],
            )
            x = torch.rand(10, channels[i], in_size[i], in_size[i])
            loss = dis(x)
            assert loss.shape == (10,)
