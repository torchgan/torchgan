import torch

from .functional import dragan_gradient_penalty
from .loss import DiscriminatorLoss, GeneratorLoss

__all__ = ["DraganGradientPenalty"]


class DraganGradientPenalty(DiscriminatorLoss):
    r"""Gradient Penalty for the DRAGAN discriminator from `"On Convergence and Stability of GANs
    by Kodali et. al." <https://arxiv.org/abs/1705.07215>`_ paper

    The gradient penalty is calculated as:

    .. math:: \lambda \times (||grad(D(x))||_2 - k)^2

    The gradient being taken with respect to x

    where

    - :math:`G` : Generator
    - :math:`D` : Disrciminator
    - :math:`\lambda` : Scaling hyperparameter
    - :math:`x` : Interpolation term for the gradient penalty
    - :math:`k` : Constant

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output.
            If ``none`` no reduction will be applied. If ``mean`` the outputs are averaged over batch size.
            If ``sum`` the elements of the output are summed.
        lambd (float,optional) : Hyperparameter :math:`\lambda` for scaling the gradient penalty.
        k (float, optional) : Constant.
        override_train_ops (function, optional): Function to be used in place of the default ``train_ops``
    """

    def __init__(self, reduction="mean", lambd=10.0, k=1.0, override_train_ops=None):
        super(DraganGradientPenalty, self).__init__(reduction)
        self.lambd = lambd
        self.override_train_ops = override_train_ops
        self.k = k

    def forward(self, interpolate, d_interpolate):
        r"""Computes the loss for the given input.

        Args:
            interpolate (torch.Tensor) : It must have the dimensions (N, \*) where
                                         \* means any number of additional dimensions.
            d_interpolate (torch.Tensor) : Output of the ``discriminator`` with ``interpolate``
                                           as the input. It must have the dimensions (N, \*)
                                           where \* means any number of additional dimensions.

        Returns:
            scalar if reduction is applied else Tensor with dimensions (N, \*).
        """
        return dragan_gradient_penalty(
            interpolate, d_interpolate, self.k, self.reduction
        )

    def train_ops(
        self,
        generator,
        discriminator,
        optimizer_discriminator,
        real_inputs,
        device,
        labels=None,
    ):
        r"""Defines the standard ``train_ops`` used by the DRAGAN Gradient Penalty.

        The ``standard optimization algorithm`` for the ``discriminator`` defined in this train_ops
        is as follows:

        1. :math:`interpolate = real + \frac{1}{2} \times (1 - \alpha) \times std(real) \times \beta`
        2. :math:`d\_interpolate = discriminator(interpolate)`
        3. :math:`loss = loss\_function(interpolate, d\_interpolate)`
        4. Backpropagate by computing :math:`\nabla loss`
        5. Run a step of the optimizer for discriminator

        Args:
            generator (torchgan.models.Generator): The model to be optimized.
            discriminator (torchgan.models.Discriminator): The discriminator which judges the
                performance of the generator.
            optimizer_discriminator (torch.optim.Optimizer): Optimizer which updates the ``parameters``
                of the ``discriminator``.
            real_inputs (torch.Tensor): The real data to be fed to the ``discriminator``.
            device (torch.device): Device on which the ``generator`` and ``discriminator`` is present.
            labels (torch.Tensor, optional): Labels for the data.

        Returns:
            Scalar value of the loss.
        """
        if self.override_train_ops is not None:
            return self.override_train_ops(
                self,
                generator,
                discriminator,
                optimizer_discriminator,
                real_inputs,
                labels,
            )
        else:
            # NOTE(avik-pal): We don't need the gradients for alpha and beta. It's there
            #                 to prevent an error while calling autograd.grad
            alpha = torch.rand(
                size=real_inputs.shape, device=device, requires_grad=True
            )
            beta = torch.rand(size=real_inputs.shape, device=device, requires_grad=True)
            optimizer_discriminator.zero_grad()
            interpolate = real_inputs + (1 - alpha) * 0.5 * real_inputs.std() * beta
            if generator.label_type == "generated":
                label_gen = torch.randint(
                    0, generator.num_classes, (real_inputs.size(0),), device=device
                )
            if discriminator.label_type == "none":
                d_interpolate = discriminator(interpolate)
            else:
                if generator.label_type == "generated":
                    d_interpolate = discriminator(interpolate, label_gen)
                else:
                    d_interpolate = discriminator(interpolate, labels)
            loss = self.forward(interpolate, d_interpolate)
            weighted_loss = self.lambd * loss
            weighted_loss.backward()
            optimizer_discriminator.step()
            return loss.item()
