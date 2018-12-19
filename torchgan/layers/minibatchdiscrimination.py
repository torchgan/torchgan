import torch
import torch.nn as nn

__all__ = ['MinibatchDiscrimination1d']

# The original paper by Salimans et. al. discusses only 1D minibatch discrimination
class MinibatchDiscrimination1d(nn.Module):
    r"""1D Minibatch Discrimination Module as proposed in the paper
    `"Improved Techniques for Training GANs by Salimans et. al." <https://arxiv.org/abs/1805.08318>`_

    Allows the Discriminator to easily detect mode collapse by augmenting the activations to the succeeding
    layer with side information that allows it to determine the 'closeness' of the minibatch examples
    with each other
        .. math :: M_i = T * \f(\vx_{i})
        .. math :: c_b(\vx_{i}, \vx_{j}) = \exp(-||M_{i, b} - M_{j, b}||_{L_1}) \in \mathbb{R}.
        .. math :: o(\vx_{i})_b &= \sum_{j=1}^{n} c_b(\vx_{i},\vx_{j}) \in \mathbb{R} \\
    o(\vx_{i}) &= \Big[ o(\vx_{i})_1, o(\vx_{i})_2, \dots, o(\vx_{i})_B \Big] \in \mathbb{R}^B \\
    o(\X) &\in \mathbb{R}^{n \times B}
    This is followed by concatenating :math: `$o(\vx_{i})$` and :math: `\f(\vx_{i})`

    Where

    - :math: `\f(\vx_{i}) \in \mathbb{R}^A` : Activations from an intermediate layer
    - :math: `\f(\vx_{i}) \in \mathbb{R}^A` : Parameter Tensor for generating minibatch discrimination matrix


    Args:
        in_features (int): Features input corresponding to dimension :math: `A`
        out_features (int): Number of output features that are to be concatenated corresponding to dimension :math: `B`
        intermediate_features (int): Intermediate number of features corresponding to dimension :math: `C`

    Returns:
        A Tensor of size :math: `(N, in_features + out_features)` where :math: `N` is the batch size
    """
    def __init__(self, in_features, out_features, intermediate_features=16):
        super(MinibatchDiscrimination1d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.intermediate_features = intermediate_features

        self.T = nn.Parameter(torch.Tensor(in_features, out_features, intermediate_features))
        nn.init.normal_(self.T)

    def forward(self, x):
        r"""Computes the output of the Minibatch Discrimination Layer

        Args:
            x (torch.Tensor): A Torch Tensor of dimensions :math: `(N, infeatures)`

        Returns:
            3D Torch Tensor of size :math: `(N,infeatures + outfeatures)` after applying Minibatch Discrimination
        """
        M = torch.mm(x, self.T.view(self.in_features, -1))
        M = M.view(-1, self.out_features, self.intermediate_features).unsqueeze(0)
        M_t = M.permute(1, 0, 2, 3)
        # Broadcasting reduces the matrix subtraction to the form desired in the paper
        out = torch.sum(torch.exp(-(torch.abs(M - M_t).sum(3))), dim=0) - 1
        return torch.cat([x, out], 1)
