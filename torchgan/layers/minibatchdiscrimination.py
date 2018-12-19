import torch
import torch.nn as nn

__all__ = ['MinibatchDiscrimination1d']

# The original paper by Salimans et. al. discusses only 1D minibatch discrimination
class MinibatchDiscrimination1d(nn.Module):
    r"""
    Minibatch Discrimination Layer as defined in Improved Training for GANs by Salimans et. al.
    The inputs should be 1 Dimensional Tensors
    """
    def __init__(self, in_features, out_features, intermediate_features=16):
        super(MinibatchDiscrimination1d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.intermediate_features = intermediate_features

        self.T = nn.Parameter(torch.Tensor(in_features, out_features, intermediate_features))
        nn.init.normal_(self.T)

    def forward(self, x):
        M = torch.mm(x, self.T.view(self.in_features, -1))
        M = M.view(-1, self.out_features, self.intermediate_features).unsqueeze(0)
        M_t = M.permute(1, 0, 2, 3)
        # Broadcasting reduces the matrix subtraction to the form desired in the paper
        out = torch.sum(torch.exp(-(torch.abs(M - M_t).sum(3))), dim=0) - 1
        return torch.cat([x, out], 1)
