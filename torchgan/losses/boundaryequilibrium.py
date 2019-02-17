import torch

from .functional import (
    boundary_equilibrium_discriminator_loss,
    boundary_equilibrium_generator_loss,
)
from .loss import DiscriminatorLoss, GeneratorLoss

__all__ = ["BoundaryEquilibriumGeneratorLoss", "BoundaryEquilibriumDiscriminatorLoss"]


class BoundaryEquilibriumGeneratorLoss(GeneratorLoss):
    r"""Boundary Equilibrium GAN generator loss from
    `"BEGAN : Boundary Equilibrium Generative Adversarial Networks
    by Berthelot et. al." <https://arxiv.org/abs/1703.10717>`_ paper

    The loss can be described as

    .. math:: L(G) = D(G(z))

    where

    - :math:`G` : Generator
    - :math:`D` : Discriminator

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output.
            If ``none`` no reduction will be applied. If ``mean`` the outputs are averaged over batch size.
            If ``sum`` the elements of the output are summed.
        override_train_ops (function, optional): Function to be used in place of the default ``train_ops``
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
        return boundary_equilibrium_generator_loss(dgz, self.reduction)


class BoundaryEquilibriumDiscriminatorLoss(DiscriminatorLoss):
    r"""Boundary Equilibrium GAN discriminator loss from
    `"BEGAN : Boundary Equilibrium Generative Adversarial Networks
    by Berthelot et. al." <https://arxiv.org/abs/1703.10717>`_ paper

    The loss can be described as

    .. math:: L(D) = D(x) - k_t \times D(G(z))

    .. math:: k_{t+1} = k_t + \lambda \times (\gamma \times D(x) - D(G(z)))

    where

    - :math:`G` : Generator
    - :math:`D` : Discriminator
    - :math:`k_t` : Running average of the balance point of G and D
    - :math:`\lambda` : Learning rate of the running average
    - :math:`\gamma` : Goal bias hyperparameter

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output.
            If ``none`` no reduction will be applied. If ``mean`` the outputs are averaged over batch size.
            If ``sum`` the elements of the output are summed.
        override_train_ops (function, optional): Function to be used in place ofthe default ``train_ops``
        init_k (float, optional): Initial value of the balance point ``k``.
        lambd (float, optional): Learning rate of the running average.
        gamma (float, optional): Goal bias hyperparameter.
    """

    def __init__(
        self,
        reduction="mean",
        override_train_ops=None,
        init_k=0.0,
        lambd=0.001,
        gamma=0.75,
    ):
        super(BoundaryEquilibriumDiscriminatorLoss, self).__init__(
            reduction, override_train_ops
        )
        self.reduction = reduction
        self.override_train_ops = override_train_ops
        self.k = init_k
        self.lambd = lambd
        self.gamma = gamma
        # TODO(Aniket1998): Integrate this with the metrics API in a later release
        self.convergence_metric = None

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
            A tuple of 3 loss values, namely the ``total loss``, ``loss due to real data`` and ``loss
            due to fake data``.
        """
        return boundary_equilibrium_discriminator_loss(dx, dgz, self.k, self.reduction)

    def set_k(self, k=0.0):
        r"""Change the default value of k

        Args:
            k (float, optional) : New value to be set.
        """
        self.k = k

    def update_k(self, loss_real, loss_fake):
        r"""Update the running mean of k for each forward pass.

        The update takes place as

        .. math:: k_{t+1} = k_t + \lambda \times (\gamma \times D(x) - D(G(z)))

        Args:
            loss_real (float): :math:`D(x)`
            loss_fake (float): :math:`D(G(z))`
        """
        diff = self.gamma * loss_real - loss_fake
        self.k += self.lambd * diff
        # TODO(Aniket1998): Develop this into a proper TorchGAN convergence metric
        self.convergence_metric = loss_real + abs(diff)
        self.k = max(min(self.k, 1.0), 0.0)

    def train_ops(
        self,
        generator,
        discriminator,
        optimizer_discriminator,
        real_inputs,
        device,
        labels=None,
    ):
        r"""Defines the standard ``train_ops`` used by boundary equilibrium loss.

        The ``standard optimization algorithm`` for the ``discriminator`` defined in this train_ops
        is as follows:

        1. :math:`fake = generator(noise)`
        2. :math:`value_1 = discriminator(fake)`
        3. :math:`value_2 = discriminator(real)`
        4. :math:`loss = loss\_function(value_1, value_2)`
        5. Backpropagate by computing :math:`\nabla loss`
        6. Run a step of the optimizer for discriminator
        7. Update the value of :math: `k`.

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
                device,
                labels,
            )
        else:
            if labels is None and (
                generator.label_type == "required"
                or discriminator.label_type == "required"
            ):
                raise Exception("GAN model requires labels for training")
            batch_size = real_inputs.size(0)
            noise = torch.randn(batch_size, generator.encoding_dims, device=device)
            if generator.label_type == "generated":
                label_gen = torch.randint(
                    0, generator.num_classes, (batch_size,), device=device
                )
            optimizer_discriminator.zero_grad()
            if discriminator.label_type == "none":
                dx = discriminator(real_inputs)
            elif discriminator.label_type == "required":
                dx = discriminator(real_inputs, labels)
            else:
                dx = discriminator(real_inputs, label_gen)
            if generator.label_type == "none":
                fake = generator(noise)
            elif generator.label_type == "required":
                fake = generator(noise, labels)
            else:
                fake = generator(noise, label_gen)
            if discriminator.label_type == "none":
                dgz = discriminator(fake.detach())
            else:
                if generator.label_type == "generated":
                    dgz = discriminator(fake.detach(), label_gen)
                else:
                    dgz = discriminator(fake.detach(), labels)
            loss_total, loss_real, loss_fake = self.forward(dx, dgz)
            loss_total.backward()
            optimizer_discriminator.step()
            self.update_k(loss_real.item(), loss_fake.item())
            return loss_total.item()
