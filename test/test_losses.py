import unittest
import torch
from sys import path
import math
path.append('..')
from torchgan.losses import *

class TestLosses(unittest.TestCase):
    def match_losses(self, l_g, l_d, dx, dgz, gen_loss_mean, gen_loss_sum, gen_loss_none,
                     d_loss_mean, d_loss_sum, d_loss_none):
        D_X = torch.Tensor(dx).view(-1, 1)
        D_GZ = torch.Tensor(dgz).view(-1, 1)

        self.assertAlmostEqual(d_loss_mean, l_d(D_X, D_GZ).item(), places=5)
        l_d.reduction = 'sum'
        self.assertAlmostEqual(d_loss_sum, l_d(D_X, D_GZ).item(), places=5)
        l_d.reduction = 'none'
        loss_none = l_d(D_X, D_GZ).view(-1, 1)
        for i in range(4):
            self.assertAlmostEqual(d_loss_none[i], loss_none[i].item(), places=5)

        self.assertAlmostEqual(gen_loss_mean, l_g(D_GZ).item(), places=5)
        l_g.reduction = 'sum'
        self.assertAlmostEqual(gen_loss_sum, l_g(D_GZ).item(), places=5)
        l_g.reduction = 'none'
        loss_none = l_g(D_GZ).view(-1, 1)
        for i in range(4):
            self.assertAlmostEqual(gen_loss_none[i], loss_none[i].item(), places=5)

    def test_wasserstein_loss(self):
        dx = [1.3, 2.9, 8.4, 6.3]
        dgz = [4.8, 1.2, -3.5, 5.9]

        gen_loss_mean = -2.1
        gen_loss_sum = -8.4
        gen_loss_none = [-4.8, -1.2, 3.5, -5.9]

        d_loss_mean = -2.625
        d_loss_sum = -10.5
        d_loss_none = [3.5000002, -1.7, -11.9, -0.4000001]

        w_g = WassersteinGeneratorLoss()
        w_d = WassersteinDiscriminatorLoss()
        self.match_losses(w_g, w_d, dx, dgz, gen_loss_mean, gen_loss_sum, gen_loss_none,
                          d_loss_mean, d_loss_sum, d_loss_none)

    def test_lsgan_loss(self):
        dx = [1.3, 2.9, 8.4, 6.3]
        dgz = [4.8, 1.2, -3.5, 5.9]

        gen_loss_mean = 7.3425007
        gen_loss_sum = 29.370003
        gen_loss_none = [7.2200007, 0.02000001, 10.125, 12.005]

        d_loss_mean = 19.76125
        d_loss_sum = 79.045
        d_loss_none = [11.565001, 2.525, 33.504997, 31.45]

        l_g = LeastSquaresGeneratorLoss()
        l_d = LeastSquaresDiscriminatorLoss()
        self.match_losses(l_g, l_d, dx, dgz, gen_loss_mean, gen_loss_sum, gen_loss_none,
                          d_loss_mean, d_loss_sum, d_loss_none)

    def test_minimax_loss(self):
        dx = [1.3, 2.9, 8.4, 6.3]
        dgz = [4.8, 1.2, -3.5, 5.9]

        factor = -torch.log(torch.sigmoid(torch.ones(1))).item()

        gen_loss_mean = -3.3642528 + factor
        gen_loss_sum = -13.457011 + 4 * factor
        gen_loss_none = [-5.1214576 + factor, -1.7765441 + factor, -0.3430121 + factor, -6.215997 + factor]

        d_loss_mean = 3.1251488
        d_loss_sum = 12.500595
        d_loss_none = [5.0492043, 1.5168452, 0.02997526, 5.90457]

        l_g = MinimaxGeneratorLoss(nonsaturating=False)
        l_d = MinimaxDiscriminatorLoss()
        self.match_losses(l_g, l_d, dx, dgz, gen_loss_mean, gen_loss_sum, gen_loss_none,
                          d_loss_mean, d_loss_sum, d_loss_none)

    def test_minimax_nonsaturating_loss(self):
        dx = [1.3, 2.9, 8.4, 6.3]
        dgz = [4.8, 1.2, -3.5, 5.9]

        gen_loss_mean = 0.9509911
        gen_loss_sum = 3.8039644
        gen_loss_none = [8.1960661e-03, 2.6328245e-01, 3.5297503e+00, 2.7356991e-03]

        d_loss_mean = 3.1251488
        d_loss_sum = 12.500595
        d_loss_none = [5.0492043, 1.5168452, 0.02997526, 5.90457]

        l_g = MinimaxGeneratorLoss(nonsaturating=True)
        l_d = MinimaxDiscriminatorLoss()
        self.match_losses(l_g, l_d, dx, dgz, gen_loss_mean, gen_loss_sum, gen_loss_none,
                          d_loss_mean, d_loss_sum, d_loss_none)
