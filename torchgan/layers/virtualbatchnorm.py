import torch
import torch.nn as nn

__all__ = ['VirtualBatchNorm']

class VirtualBatchNorm(nn.Module):
    def __init__(self, in_features, eps=1e-5):
        super(VirtualBatchNorm, self).__init__()
        self.in_features = in_features
        self.scale = nn.Parameter(torch.ones(in_features))
        self.bias = nn.Parameter(torch.zeros(in_features))
        self.ref_mu = None
        self.ref_var = None
        self.eps = eps

    def _batch_stats(self, x):
        mu = torch.mean(x, dim=0, keepdim=True)
        var = torch.var(x, dim=0, keepdim=True)
        return mu, var

    def _normalize(self, x, mu, var):
        std = torch.sqrt(self.eps + var)
        x = (x - mu) / std
        sizes = list(x.size())
        for dim, i in enumerate(x.size()):
            if dim != 1:
                sizes[dim] = 1
        scale = self.scale.view(*sizes)
        bias = self.bias.view(*sizes)
        return x * scale + bias

    def forward(self, x):
        assert x.size(1) == self.in_features
        if self.ref_mu is None or self.ref_var is None:
            self.ref_mu, self.ref_var = self._batch_stats(x)
            self.ref_mu = self.ref_mu.clone().detach()
            self.ref_var = self.ref_var.clone().detach()
            out = self._normalize(x, self.ref_mu, self.ref_var)
        else:
            out = self._normalize(x, self.ref_mu, self.ref_var)
            self.ref_mu = None
            self.ref_var = None
        return out
