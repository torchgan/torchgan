import torch
import torch.nn as nn
from torch.nn import Parameter

__all__ = ["SpectralNorm2d"]

# NOTE(avik-pal): This code has been adapted from
#                 https://github.com/heykeetae/Self-Attention-GAN/blob/master/spectral.py
class SpectralNorm2d(nn.Module):
    r"""2D Spectral Norm Module as described in `"Spectral Normalization
    for Generative Adversarial Networks by Miyato et. al." <https://arxiv.org/abs/1802.05957>`_
    The spectral norm is computed using ``power iterations``.

    Computation Steps:

    .. math:: v_{t + 1} = \frac{W^T W v_t}{||W^T W v_t||} = \frac{(W^T W)^t v}{||(W^T W)^t v||}
    .. math:: u_{t + 1} = W v_t
    .. math:: v_{t + 1} = W^T u_{t + 1}
    .. math:: Norm(W) = ||W v|| = u^T W v
    .. math:: Output = \frac{W}{Norm(W)} = \frac{W}{u^T W v}

    Args:
        module (torch.nn.Module): The Module on which the Spectral Normalization needs to be
            applied.
        name (str, optional): The attribute of the ``module`` on which normalization needs to
            be performed.
        power_iterations (int, optional): Total number of iterations for the norm to converge.
            ``1`` is usually enough given the weights vary quite gradually.

    Example:
        .. code:: python

            >>> layer = SpectralNorm2d(Conv2d(3, 16, 1))
            >>> x = torch.rand(1, 3, 10, 10)
            >>> layer(x)
    """

    def __init__(self, module, name="weight", power_iterations=1):
        super(SpectralNorm2d, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        self.u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        self.v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        self.u.data = self._l2normalize(self.u.data)
        self.v.data = self._l2normalize(self.v.data)
        self.w_bar = Parameter(w.data)
        del self.module._parameters[self.name]

    def _l2normalize(self, x, eps=1e-12):
        r"""Function to calculate the ``L2 Normalized`` form of a Tensor

        Args:
            x (torch.Tensor): Tensor which needs to be normalized.
            eps (float, optional): A small value needed to avoid infinite values.

        Returns:
            Normalized form of the tensor ``x``.
        """
        return x / (torch.norm(x) + eps)

    def forward(self, *args):
        r"""Computes the output of the ``module`` and appies spectral normalization to the
        ``name`` attribute of the ``module``.

        Returns:
            The output of the ``module``.
        """
        height = self.w_bar.data.shape[0]
        for _ in range(self.power_iterations):
            self.v.data = self._l2normalize(
                torch.mv(torch.t(self.w_bar.view(height, -1)), self.u)
            )
            self.u.data = self._l2normalize(
                torch.mv(self.w_bar.view(height, -1), self.v)
            )
        sigma = self.u.dot(self.w_bar.view(height, -1).mv(self.v))
        setattr(self.module, self.name, self.w_bar / sigma.expand_as(self.w_bar))
        return self.module.forward(*args)
