import torch
import torch.autograd as autograd
from .loss import GeneratorLoss, DiscriminatorLoss
from ..utils import reduce

__all__ = ['dragan_gradient_penalty', 'DraganGradientPenalty']

def dragan_gradient_penalty(interpolate, d_interpolate, k=1.0, reduction='elementwise_mean'):
    grad_outputs = torch.ones_like(d_interpolate)
    gradients = autograd.grad(outputs=d_interpolate, inputs=interpolate,
                              grad_outputs=grad_outputs,
                              create_graph=True, retain_graph=True,
                              only_inputs=True, allow_unused=True)[0]

    gradient_penalty = (gradients.norm(2) - k) ** 2
    return reduce(gradient_penalty, reduction)

class DraganGradientPenalty(DiscriminatorLoss):
    r"""Gradient Penalty for the DRAGAN discriminator from
    `"On Convergence and Stability of GANs
    by Kodali et. al." <https://arxiv.org/abs/1705.07215>`_ paper

    The gradient penalty is calculated as:

    .. math: \lambda \times (norm(grad(D(x))) - 1)^2

    The gradient being taken with respect to x

    where

    - G : Generator
    - D : Disrciminator
    - :math:`\lambda` : Scaling hyperparameter (default 10.0)
    - x : Interpolation term for the gradient penalty

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output.
            If `none` no reduction will be applied. If `elementwise_mean` the sum of
            the elements will be divided by the number of elements in the output. If
            `sum` the output will be summed.

        lambd(float,optional) : Hyperparameter lambda for scaling the gradient penalty.
    """
    def __init__(self, reduction='elementwise_mean', lambd=10.0, k=1.0, override_train_ops=None):
        super(DraganGradientPenalty, self).__init__(reduction)
        self.lambd = lambd
        self.override_train_ops = override_train_ops
        self.k = k

    def forward(self, interpolate, d_interpolate):
        r"""
        Args:
            interpolate (torch.Tensor) : It must have the dimensions (N, \*) where
                                         \* means any number of additional dimensions.
            d_interpolate (torch.Tensor) : It must have the dimensions (N, \*) where
                                           \* means any number of additional dimensions.

        Returns:
            scalar if reduction is applied else Tensor with dimensions (N, \*).
        """
        return dragan_gradient_penalty(interpolate, d_interpolate, self.k, self.reduction)

    def train_ops(self, generator, discriminator, optimizer_discriminator, real_inputs,
                  device, labels=None):
        if self.override_train_ops is not None:
            return self.override_train_ops(self, generator, discriminator, optimizer_discriminator,
                                           real_inputs, labels)
        else:
            # NOTE(avik-pal): We don't need the gradients for alpha and beta. Its there
            #                 to prevent an error while calling autograd.grad
            alpha = torch.rand(size=real_inputs.shape, device=device, requires_grad=True)
            beta = torch.rand(size=real_inputs.shape, device=device, requires_grad=True)
            optimizer_discriminator.zero_grad()
            interpolate = real_inputs + (1 - alpha) * 0.5 * real_inputs.std() * beta
            if generator.label_type == 'generated':
                label_gen = torch.randint(0, generator.num_classes, (real_inputs.size(0),), device=device)
            if discriminator.label_type == 'none':
                d_interpolate = discriminator(interpolate)
            else:
                if generator.label_type == 'generated':
                    d_interpolate = discriminator(interpolate, label_gen)
                else:
                    d_interpolate = discriminator(interpolate, labels)
            loss = self.forward(interpolate, d_interpolate)
            weighted_loss = self.lambd * loss
            weighted_loss.backward()
            optimizer_discriminator.step()
            return loss.item()
