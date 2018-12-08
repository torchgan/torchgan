import torch
import torch.nn.functional as F
from .loss import GeneratorLoss, DiscriminatorLoss
from ..utils import reduce

__all__ = ['energy_based_generator_loss', 'energy_based_discriminator_loss',
           'EnergyBasedGeneratorLoss', 'EnergyBasedDiscriminatorLoss']

def energy_based_generator_loss(dgz, reduction):
    return reduce(dgz, reduction)

def energy_based_discriminator_loss(dx, dgz, margin, reduction='mean'):
    return reduce(dx + F.relu(-dgz + margin), reduction)

class EnergyBasedGeneratorLoss(GeneratorLoss):
    r"""Energy Based GAN generator loss from `"Energy Based Generative Adversarial Network
    by Zhao et. al." <https://arxiv.org/abs/1609.03126>`_ paper.

    The loss can be described as:

    .. math:: L(G) = D(G(z))

    where

    - :math:`G` : Generator
    - :math:`D` : Discriminator
    - :math:`z` : A sample from the noise prior

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output.
            If ``none`` no reduction will be applied. If ``mean`` the mean of the output.
            If ``sum`` the elements of the output will be summed.
        override_train_ops (function, optional): A function is passed to this argument,
            if the default ``train_ops`` is not to be used.
    """
    def forward(self, dgz):
        r"""Computes the loss for the given input.

        Args:
            dgz (torch.Tensor) : Output of the Discriminator with generated data. It must have the
                                 dimensions (N, \*) where \* means any number of additional
                                 dimensions.

        Returns:
            scalar if reduction is applied else Tensor with dimensions (N, \*).
        """
        return energy_based_generator_loss(dgz, self.reduction)


class EnergyBasedDiscriminatorLoss(DiscriminatorLoss):
    r"""Energy Based GAN generator loss from `"Energy Based Generative Adversarial Network
    by Zhao et. al." <https://arxiv.org/abs/1609.03126>`_ paper

    The loss can be described as:

    .. math:: L(D) = D(x) + max(0, m - D(G(z)))

    where

    - :math:`G` : Generator
    - :math:`D` : Discriminator
    - :math:`m` : Margin Hyperparameter
    - :math:`z` : A sample from the noise prior

    .. note::
        The convergence of EBGAN is highly sensitive to hyperparameters. The ``margin``
        hyperparameter as per the paper was taken as follows:

        +----------------------+--------+
        | Dataset              | Margin |
        +======================+========+
        | MNIST                | 10.0   |
        +----------------------+--------+
        | LSUN                 | 80.0   |
        +----------------------+--------+
        | CELEB A              | 20.0   |
        +----------------------+--------+
        | Imagenet (128 x 128) | 40.0   |
        +----------------------+--------+
        | Imagenet (256 x 256) | 80.0   |
        +----------------------+--------+

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output.
            If ``none`` no reduction will be applied. If ``mean`` the mean of the output.
            If ``sum`` the elements of the output will be summed.
        margin (float, optional): The margin hyperparameter.
        override_train_ops (function, optional): A function is passed to this argument,
            if the default ``train_ops`` is not to be used.
    """
    def __init__(self, reduction='mean', margin=80.0, override_train_ops=None):
        super(EnergyBasedDiscriminatorLoss, self).__init__(reduction, override_train_ops)
        self.margin = margin

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
        return reduce(dx + F.relu(-dgz + self.margin), self.reduction)
