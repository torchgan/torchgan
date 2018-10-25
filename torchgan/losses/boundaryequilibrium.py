import torch
from .loss import GeneratorLoss, DiscriminatorLoss
from ..utils import reduce

__all__ = ['BoundaryEquilibriumGeneratorLoss', 'BoundaryEquilibriumDiscriminatorLoss']

class BoundaryEquilibriumGeneratorLoss(GeneratorLoss):
    r"""Boundary Equilibrium GAN generator loss from
    `"BEGAN : Boundary Equilibrium Generative Adversarial Networks
    by Berthelot et. al." <https://arxiv.org/abs/1703.10717>`_ paper

    The loss can be described as

    .. math:: L(G) = D(G(z))

    where

    - G : Generator
    - D : Discriminator

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output.
            If `none` no reduction will be applied. If `elementwise_mean` the sum of
            the elements will be divided by the number of elements in the output. If
            `sum` the output will be summed.
    """
    def forward(self, dgz):
        return reduce(dgz, self.reduction)

class BoundaryEquilibriumDiscriminatorLoss(DiscriminatorLoss):
    r"""Boundary Equilibrium GAN discriminator loss from
    `"BEGAN : Boundary Equilibrium Generative Adversarial Networks
    by Berthelot et. al." <https://arxiv.org/abs/1703.10717>`_ paper

    The loss can be described as

    .. math:: L(D) = D(x) - k_t \times D(G(z))

    .. math:: k_{t+1} = k_t + \lambda \times (\gamma \times D(x) - D(G(z)))

    where

    - G : Generator
    - D : Discriminator
    - :math:`k_t` : Running average of the balance point of G and D
    - :math:`\lambda` : Learning rate of the running average
    - :math:`\gamma` : Goal bias hyperparameter

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output.
            If `none` no reduction will be applied. If `elementwise_mean` the sum of
            the elements will be divided by the number of elements in the output. If
            `sum` the output will be summed.
    """
    def __init__(self, reduction='elementwise_mean', override_train_ops=None, init_k=0.0, lambd=0.001, gamma=0.75):
        self.reduction = reduction
        self.override_train_ops = override_train_ops
        self.k = init_k
        self.lambd = lambd
        self.gamma = gamma
        self.convergence_metric = None

    def forward(self, dx, dgz):
        r"""
        Args:
            dx (torch.Tensor) : Output of the Discriminator. It must have the dimensions
                                (N, \*) where \* means any number of additional dimensions.
            dgz (torch.Tensor) : Output of the Generator. It must have the dimensions
                                 (N, \*) where \* means any number of additional dimensions.
        Returns:
            scalar tuple if reduction is applied else Tensor tuple each with dimensions (N, \*).
        """
        loss_real = reduce(dx, self.reduction)
        loss_fake = reduce(dgz, self.reduction)
        loss_total = loss_real - self.k * loss_fake
        return loss_total, loss_real, loss_fake

    def set_k(self, k=0.0):
        r"""Change the default value of k

        Args:
            k (float, optional) : New value to be set.
        """
        self.k = k

    def update_k(self, loss_real, loss_fake):
        r"""Update the running mean of k for each forward pass
        The update takes place as
        .. math:: k_{t+1} = k_t + \lambda \times (\gamma \times D(x) - D(G(z)))

        Args:
            loss_real: :math:`D(x)`
            loss_fake: :math:`D(G(z))`
        """
        diff = self.gamma * loss_real - loss_fake
        self.k += self.lambd * diff
        # TODO(Aniket1998): Develop this into a proper TorchGAN convergence metric
        self.convergence_metric = loss_real + abs(diff)
        if self.k < 0.0:
            self.k = 0.0
        elif self.k > 1.0:
            self.k = 1.0

    def train_ops(self, generator, discriminator, optimizer_discriminator, real_inputs, device, labels_provided=False):
        if self.override_train_ops is not None:
            return self.override_train_ops(generator, discriminator, optimizer_discriminator,
                                           real_inputs, device, labels_provided)
        else:
            real = real_inputs if labels_provided is False else real_inputs[0]
            noise = torch.randn(real.size(0), generator.encoding_dims, device=device)
            optimizer_discriminator.zero_grad()
            dx = discriminator(real)
            fake = generator(noise)
            dgz = discriminator(fake.detach())
            loss_total, loss_real, loss_fake = self.forward(dx, dgz)
            loss_total.backward()
            optimizer_discriminator.step()
            self.update_k(loss_real.item(), loss_fake.item())
            return loss_total.item()
