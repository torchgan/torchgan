import torch
import torch.nn as nn

__all__ = ['VirtualBatchNorm']

class VirtualBatchNorm(nn.Module):
    def __init__(self, in_features, ref_batch, eps=1e-5):
        super(VirtualBatchNorm, self).__init__()
        self.in_features = in_features
        sizes = [1 for i in range(len(ref_batch.shape))]
        sizes[1] = ref_batch.size(1)
        self.weight = nn.Parameter(torch.Tensor(*sizes))
        self.bias = nn.Parameter(torch.Tensor(*sizes))
        # TODO(Aniket1998): Error occurs if input batch size does not match reference batch size
        # which might not be enforced during test time. The original paper does not address this either
        self.ref_batch = ref_batch
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
        ref_mu, ref_var = self._batch_stats(self.ref_batch)
        ref_mu = ref_mu.clone().detach()
        ref_var = ref_var.clone().detach()
        # During test time, the optimal weight and bias are adjusted according to the statistics
        # of the reference batch, which should then not change
        if self.training:
            self.ref_batch = self._normalize(self.ref_batch, ref_mu, ref_var)
        return self._normalize(x, ref_mu, ref_var)
