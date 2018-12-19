import torch
import torch.nn.functional as F
from .loss import GeneratorLoss, DiscriminatorLoss

__all__ = ['minimax_generator_loss', 'minimax_discriminator_loss', 'MinimaxGeneratorLoss', 'MinimaxDiscriminatorLoss']

def minimax_generator_loss(dgz, nonsaturating=True, reduction='mean'):
    if nonsaturating:
        target = torch.ones_like(dgz)
        return F.binary_cross_entropy_with_logits(dgz, target,
                                                  reduction=reduction)
    else:
        target = torch.zeros_like(dgz)
        return -1.0 * F.binary_cross_entropy_with_logits(dgz, target,
                                                         reduction=reduction)

def minimax_discriminator_loss(dx, dgz, label_smoothing=0.0, reduction='elementwise_mean'):
    target_ones = torch.ones_like(dgz) * (1.0 - label_smoothing)
    target_zeros = torch.zeros_like(dx)
    loss = F.binary_cross_entropy_with_logits(dx, target_ones,
                                              reduction=reduction)
    loss += F.binary_cross_entropy_with_logits(dgz, target_zeros,
                                               reduction=reduction)
    return loss

class MinimaxGeneratorLoss(GeneratorLoss):
    r"""Minimax game generator loss from the original GAN paper `"Generative Adversarial Networks
    by Goodfellow et. al." <https://arxiv.org/abs/1406.2661>`_

    The loss can be described as:

    .. math:: L(G) = log(1 - D(G(z)))

    The nonsaturating heuristic is also supported:

    .. math:: L(G) = -log(D(G(z)))

    where

    - :math:`G` : Generator
    - :math:`D` : Discriminator
    - :math:`z` : A sample from the noise prior

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output.
            If ``none`` no reduction will be applied. If ``mean`` the outputs are averaged over batch size.
            If ``sum`` the elements of the output are summed.
        override_train_ops (function, optional): Function to be used in place of the default ``train_ops``
        nonsaturating(bool, optional): Specifies whether to use the nonsaturating heuristic
            loss for the generator.
    """
    def __init__(self, reduction='mean', nonsaturating=True, override_train_ops=None):
        super(MinimaxGeneratorLoss, self).__init__(reduction, override_train_ops)
        self.nonsaturating = nonsaturating

    def forward(self, dgz):
        r"""Computes the loss for the given input.

        Args:
            dgz (torch.Tensor) : Output of the Discriminator with generated data. It must have the
                                 dimensions (N, \*) where \* means any number of additional
                                 dimensions.

        Returns:
            scalar if reduction is applied else Tensor with dimensions (N, \*).
        """
        return minimax_generator_loss(dgz, self.nonsaturating, self.reduction)


class MinimaxDiscriminatorLoss(DiscriminatorLoss):
    r"""Minimax game discriminator loss from the original GAN paper `"Generative Adversarial Networks
    by Goodfellow et. al." <https://arxiv.org/abs/1406.2661>`_

    The loss can be described as:

    .. math:: L(D) = -[log(D(x)) + log(1 - D(G(z)))]

    where

    - :math:`G` : Generator
    - :math:`D` : Discriminator
    - :math:`x` : A sample from the data distribution
    - :math:`z` : A sample from the noise prior

    Args:
        label_smoothing (float, optional): The factor by which the labels (1 in this case) needs
            to be smoothened. For example, label_smoothing = 0.2 changes the value of the real
            labels to 0.8.
        reduction (str, optional): Specifies the reduction to apply to the output.
            If ``none`` no reduction will be applied. If ``mean`` the mean of the output.
            If ``sum`` the elements of the output will be summed.
        override_train_ops (function, optional): A function is passed to this argument,
            if the default ``train_ops`` is not to be used.
    """
    def __init__(self, label_smoothing=0.0, reduction='mean', override_train_ops=None):
        super(MinimaxDiscriminatorLoss, self).__init__(reduction, override_train_ops)
        self.label_smoothing = label_smoothing

    def forward(self, dx, dgz):
        r"""Computes the loss for the given input.

        Args:
            dx (torch.Tensor) : Output of the Discriminator with real data. It must have the
                                dimensions (N, \*) where \* means any number of additional
                                dimensions.
            dgz (torch.Tensor) : Output of the Discriminator with generated data. It must have the
                                 dimensions (N, \*) where \* means any number of additional
                                 dimensions.

        Returns:
            scalar if reduction is applied else Tensor with dimensions (N, \*).
        """
        return minimax_discriminator_loss(dx, dgz, label_smoothing=self.label_smoothing, reduction=self.reduction)
