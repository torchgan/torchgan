import torch
import torch.nn as nn

__all__ = ['VirtualBatchNorm']

class VirtualBatchNorm(nn.Module):
    def __init__(self, in_features, dims=2, eps=1e-5):
        super(VirtualBatchNorm, self).__init__()
        self.in_features = in_features
        if dims == 1:
            sizes = [1 for _ in range(dims + 1)]  # 1D inputs will be of the form (N,C)
        else:
            sizes = [1 for _ in range(dims + 2)]  # 2D inputs are of the form (N,C,X,Y)
        self.weight = nn.Parameter(torch.Tensor(*sizes))
        self.bias = nn.Parameter(torch.Tensor(*sizes))
        self.ref_mu = None
        self.ref_var = None
        self.eps = eps

    def _batch_stats(self, x):
        mu = torch.mean(x, dim=0, keepdim=True)
        mu_2 = torch.mean(x ** 2, dim=0, keepdim=True)
        var = mu_2 - mu ** 2
        return mu, var

    def _normalize(self, x, mu, var):
        std = torch.sqrt(self.eps + var)
        x = (x - mu) / std
        return x * self.weight + self.bias

    def forward(self, x):
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
