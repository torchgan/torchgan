import torch

from .functional import least_squares_discriminator_loss, least_squares_generator_loss
from .loss import DiscriminatorLoss, GeneratorLoss

__all__ = ["LeastSquaresGeneratorLoss", "LeastSquaresDiscriminatorLoss"]


class LeastSquaresGeneratorLoss(GeneratorLoss):
    r"""Least Squares GAN generator loss from `"Least Squares Generative Adversarial Networks
    by Mao et. al." <https://arxiv.org/abs/1611.04076>`_ paper

    The loss can be described as

    .. math:: L(G) = \frac{(D(G(z)) - c)^2}{2}

    where

    - :math:`G` : Generator
    - :math:`D` : Disrciminator
    - :math:`c` : target generator label
    - :math:`z` : A sample from the noise prior

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output.
            If ``none`` no reduction will be applied. If ``mean`` the outputs are averaged over batch size.
            If ``sum`` the elements of the output are summed.
        c (float, optional): Target generator label.
        override_train_ops (function, optional): Function to be used in place of the default ``train_ops``
    """

    def __init__(self, reduction="mean", c=1.0, override_train_ops=None):
        super(LeastSquaresGeneratorLoss, self).__init__(reduction, override_train_ops)
        self.c = c

    def forward(self, dgz):
        r"""Computes the loss for the given input.

        Args:
            dgz (torch.Tensor) : Output of the Discriminator with generated data. It must have the
                                 dimensions (N, \*) where \* means any number of additional
                                 dimensions.

        Returns:
            scalar if reduction is applied else Tensor with dimensions (N, \*).
        """
        return least_squares_generator_loss(dgz, self.c, self.reduction)


class LeastSquaresDiscriminatorLoss(DiscriminatorLoss):
    r"""Least Squares GAN discriminator loss from `"Least Squares Generative Adversarial Networks
    by Mao et. al." <https://arxiv.org/abs/1611.04076>`_ paper.

    The loss can be described as:

    .. math:: L(D) = \frac{(D(x) - b)^2 + (D(G(z)) - a)^2}{2}

    where

    - :math:`G` : Generator
    - :math:`D` : Disrciminator
    - :math:`a` : Target discriminator label for generated image
    - :math:`b` : Target discriminator label for real image

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output.
            If ``none`` no reduction will be applied. If ``mean`` the outputs are averaged over batch size.
            If ``sum`` the elements of the output are summed.
        a (float, optional): Target discriminator label for generated image.
        b (float, optional): Target discriminator label for real image.
        override_train_ops (function, optional): Function to be used in place of the default ``train_ops``
    """

    def __init__(self, reduction="mean", a=0.0, b=1.0, override_train_ops=None):
        super(LeastSquaresDiscriminatorLoss, self).__init__(
            reduction, override_train_ops
        )
        self.a = a
        self.b = b

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
        return least_squares_discriminator_loss(dx, dgz, self.a, self.b, self.reduction)
