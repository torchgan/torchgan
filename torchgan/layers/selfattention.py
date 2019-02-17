import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SelfAttention2d"]


class SelfAttention2d(nn.Module):
    r"""Self Attention Module as proposed in the paper `"Self-Attention Generative Adversarial
    Networks by Han Zhang et. al." <https://arxiv.org/abs/1805.08318>`_

    .. math:: attention = softmax((query(x))^T * key(x))
    .. math:: output = \gamma * value(x) * attention + x

    where

    - :math:`query` : 2D Convolution Operation
    - :math:`key` : 2D Convolution Operation
    - :math:`value` : 2D Convolution Operation
    - :math:`x` : Input

    Args:
        input_dims (int): The input channel dimension in the input ``x``.
        output_dims (int, optional): The output channel dimension. If ``None`` the output
            channel value is computed as ``input_dims // 8``. So if the ``input_dims`` is **less
            than 8** then the layer will give an error.
        return_attn (bool, optional): Set it to ``True`` if you want the attention values to be
            returned.
    """

    def __init__(self, input_dims, output_dims=None, return_attn=False):
        output_dims = input_dims // 8 if output_dims is None else output_dims
        if output_dims == 0:
            raise Exception(
                "The output dims corresponding to the input dims is 0. Increase the input\
                            dims to 8 or more. Else specify output_dims"
            )
        super(SelfAttention2d, self).__init__()
        self.query = nn.Conv2d(input_dims, output_dims, 1)
        self.key = nn.Conv2d(input_dims, output_dims, 1)
        self.value = nn.Conv2d(input_dims, input_dims, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.return_attn = return_attn

    def forward(self, x):
        r"""Computes the output of the Self Attention Layer

        Args:
            x (torch.Tensor): A 4D Tensor with the channel dimension same as ``input_dims``.

        Returns:
            A tuple of the ``output`` and the ``attention`` if ``return_attn`` is set to ``True``
            else just the ``output`` tensor.
        """
        dims = (x.size(0), -1, x.size(2) * x.size(3))
        out_query = self.query(x).view(dims)
        out_key = self.key(x).view(dims).permute(0, 2, 1)
        attn = F.softmax(torch.bmm(out_key, out_query), dim=-1)
        out_value = self.value(x).view(dims)
        out_value = torch.bmm(out_value, attn).view(x.size())
        out = self.gamma * out_value + x
        if self.return_attn:
            return out, attn
        return out
