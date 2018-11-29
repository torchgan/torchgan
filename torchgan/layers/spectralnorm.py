import torch
import torch.nn as nn
from torch.nn import Parameter

__all__ = ['SpectralNorm']

# NOTE(avik-pal): This code has been adapted from
#                 https://github.com/heykeetae/Self-Attention-GAN/blob/master/spectral.py
class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        self.u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        self.v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        self.u.data = self.l2normalize(self.u.data)
        self.v.data = self.l2normalize(self.v.data)
        self.w_bar = Parameter(w.data)
        del self.module._parameters[self.name]

    def l2normalize(self, x, eps=1e-12):
        return x / (torch.norm(x) + eps)

    def forward(self, *args):
        height = self.w_bar.data.shape[0]
        for _ in range(self.power_iterations):
            self.v.data = self.l2normalize(torch.mv(torch.t(self.w_bar.view(height, -1)), self.u))
            self.u.data = self.l2normalize(torch.mv(self.w_bar.view(height, -1), self.v))
        sigma = self.u.dot(self.w_bar.view(height, -1).mv(self.v))
        setattr(self.module, self.name, self.w_bar / sigma.expand_as(self.w_bar))
        return self.module.forward(*args)
