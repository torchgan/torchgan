import torch
import torch.nn as nn

__all__ = ["VirtualBatchNorm"]


class VirtualBatchNorm(nn.Module):
    r"""Virtual Batch Normalization Module as proposed in the paper
    `"Improved Techniques for Training GANs by Salimans et. al." <https://arxiv.org/abs/1805.08318>`_

    Performs Normalizes the features of a batch based on the statistics collected on a reference
    batch of samples that are chosen once and fixed from the start, as opposed to regular
    batch normalization that uses the statistics of the batch being normalized

    Virtual Batch Normalization requires that the size of the batch being normalized is at least
    a multiple of (and ideally equal to) the size of the reference batch. Keep this in mind while
    choosing the batch size in ```torch.utils.data.DataLoader``` or use ```drop_last=True```

    .. math:: y = \frac{x - \mathrm{E}[x_{ref}]}{\sqrt{\mathrm{Var}[x_{ref}] + \epsilon}} * \gamma + \beta

    where

    - :math:`x` : Batch Being Normalized
    - :math:`x_{ref}` : Reference Batch

    Args:
        in_features (int): Size of the input dimension to be normalized
        eps (float, optional): Value to be added to variance for numerical stability while normalizing
    """

    def __init__(self, in_features, eps=1e-5):
        super(VirtualBatchNorm, self).__init__()
        self.in_features = in_features
        self.scale = nn.Parameter(torch.ones(in_features))
        self.bias = nn.Parameter(torch.zeros(in_features))
        self.ref_mu = None
        self.ref_var = None
        self.eps = eps

    def _batch_stats(self, x):
        r"""Computes the statistics of the batch ``x``.

        Args:
            x (torch.Tensor): Tensor whose statistics need to be computed.

        Returns:
            A tuple of the mean and variance of the batch ``x``.
        """
        mu = torch.mean(x, dim=0, keepdim=True)
        var = torch.var(x, dim=0, keepdim=True)
        return mu, var

    def _normalize(self, x, mu, var):
        r"""Normalizes the tensor ``x`` using the statistics ``mu`` and ``var``.

        Args:
            x (torch.Tensor): The Tensor to be normalized.
            mu (torch.Tensor): Mean using which the Tensor is to be normalized.
            var (torch.Tensor): Variance used in the normalization of ``x``.

        Returns:
            Normalized Tensor ``x``.
        """
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
        r"""Computes the output of the Virtual Batch Normalization

        Args:
            x (torch.Tensor): A Torch Tensor of dimension at least 2 which is to be Normalized

        Returns:
            Torch Tensor of the same dimension after normalizing with respect to the statistics of the reference batch
        """
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
