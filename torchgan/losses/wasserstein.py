import torch

from .functional import (
    wasserstein_discriminator_loss,
    wasserstein_generator_loss,
    wasserstein_gradient_penalty,
)
from .loss import DiscriminatorLoss, GeneratorLoss

__all__ = [
    "WassersteinGeneratorLoss",
    "WassersteinDiscriminatorLoss",
    "WassersteinGradientPenalty",
]


class WassersteinGeneratorLoss(GeneratorLoss):
    r"""Wasserstein GAN generator loss from
    `"Wasserstein GAN by Arjovsky et. al." <https://arxiv.org/abs/1701.07875>`_ paper

    The loss can be described as:

    .. math:: L(G) = -f(G(z))

    where

    - :math:`G` : Generator
    - :math:`f` : Critic/Discriminator
    - :math:`z` : A sample from the noise prior

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output.
            If ``none`` no reduction will be applied. If ``mean`` the mean of the output.
            If ``sum`` the elements of the output will be summed.
        override_train_ops (function, optional): A function is passed to this argument,
            if the default ``train_ops`` is not to be used.
    """

    def forward(self, fgz):
        r"""Computes the loss for the given input.

        Args:
            dgz (torch.Tensor) : Output of the Discriminator with generated data. It must have the
                                 dimensions (N, \*) where \* means any number of additional
                                 dimensions.

        Returns:
            scalar if reduction is applied else Tensor with dimensions (N, \*).
        """
        return wasserstein_generator_loss(fgz, self.reduction)


class WassersteinDiscriminatorLoss(DiscriminatorLoss):
    r"""Wasserstein GAN generator loss from
    `"Wasserstein GAN by Arjovsky et. al." <https://arxiv.org/abs/1701.07875>`_ paper

    The loss can be described as:

    .. math:: L(D) = f(G(z)) - f(x)

    where

    - :math:`G` : Generator
    - :math:`f` : Critic/Discriminator
    - :math:`x` : A sample from the data distribution
    - :math:`z` : A sample from the noise prior

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output.
            If ``none`` no reduction will be applied. If ``mean`` the mean of the output.
            If ``sum`` the elements of the output will be summed.
        clip (tuple, optional): Tuple that specifies the maximum and minimum parameter
            clamping to be applied, as per the original version of the Wasserstein loss
            without Gradient Penalty.
        override_train_ops (function, optional): A function is passed to this argument,
            if the default ``train_ops`` is not to be used.
    """

    def __init__(self, reduction="mean", clip=None, override_train_ops=None):
        super(WassersteinDiscriminatorLoss, self).__init__(
            reduction, override_train_ops
        )
        if (isinstance(clip, tuple) or isinstance(clip, list)) and len(clip) > 1:
            self.clip = clip
        else:
            self.clip = None

    def forward(self, fx, fgz):
        r"""Computes the loss for the given input.

        Args:
            fx (torch.Tensor) : Output of the Discriminator with real data. It must have the
                                dimensions (N, \*) where \* means any number of additional
                                dimensions.
            fgz (torch.Tensor) : Output of the Discriminator with generated data. It must have the
                                 dimensions (N, \*) where \* means any number of additional
                                 dimensions.

        Returns:
            scalar if reduction is applied else Tensor with dimensions (N, \*).
        """
        return wasserstein_discriminator_loss(fx, fgz, self.reduction)

    def train_ops(
        self,
        generator,
        discriminator,
        optimizer_discriminator,
        real_inputs,
        device,
        labels=None,
    ):
        r"""Defines the standard ``train_ops`` used by wasserstein discriminator loss.

        The ``standard optimization algorithm`` for the ``discriminator`` defined in this train_ops
        is as follows:

        1. Clamp the discriminator parameters to satisfy :math:`lipschitz\ condition`
        2. :math:`fake = generator(noise)`
        3. :math:`value_1 = discriminator(fake)`
        4. :math:`value_2 = discriminator(real)`
        5. :math:`loss = loss\_function(value_1, value_2)`
        6. Backpropagate by computing :math:`\nabla loss`
        7. Run a step of the optimizer for discriminator

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
                generator,
                discriminator,
                optimizer_discriminator,
                real_inputs,
                device,
                labels,
            )
        else:
            if self.clip is not None:
                for p in discriminator.parameters():
                    p.data.clamp_(self.clip[0], self.clip[1])
            return super(WassersteinDiscriminatorLoss, self).train_ops(
                generator,
                discriminator,
                optimizer_discriminator,
                real_inputs,
                device,
                labels,
            )


class WassersteinGradientPenalty(DiscriminatorLoss):
    r"""Gradient Penalty for the Improved Wasserstein GAN discriminator from
    `"Improved Training of Wasserstein GANs
    by Gulrajani et. al." <https://arxiv.org/abs/1704.00028>`_ paper

    The gradient penalty is calculated as:

    .. math: \lambda \times (||\nabla(D(x))||_2 - 1)^2

    The gradient being taken with respect to x

    where

    - :math:`G` : Generator
    - :math:`D` : Disrciminator/Critic
    - :math:`\lambda` : Scaling hyperparameter
    - :math:`x` : Interpolation term for the gradient penalty

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output.
            If ``none`` no reduction will be applied. If ``mean`` the mean of the output.
            If ``sum`` the elements of the output will be summed.
        lambd (float,optional): Hyperparameter lambda for scaling the gradient penalty.
        override_train_ops (function, optional): A function is passed to this argument,
            if the default ``train_ops`` is not to be used.
    """

    def __init__(self, reduction="mean", lambd=10.0, override_train_ops=None):
        super(WassersteinGradientPenalty, self).__init__(reduction, override_train_ops)
        self.lambd = lambd
        self.override_train_ops = override_train_ops

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
        # TODO(Aniket1998): Check for performance bottlenecks
        # If found, write the backprop yourself instead of
        # relying on autograd
        return wasserstein_gradient_penalty(interpolate, d_interpolate, self.reduction)

    def train_ops(
        self,
        generator,
        discriminator,
        optimizer_discriminator,
        real_inputs,
        device,
        labels=None,
    ):
        r"""Defines the standard ``train_ops`` used by the Wasserstein Gradient Penalty.

        The ``standard optimization algorithm`` for the ``discriminator`` defined in this train_ops
        is as follows:

        1. :math:`fake = generator(noise)`
        2. :math:`interpolate = \epsilon \times real + (1 - \epsilon) \times fake`
        3. :math:`d\_interpolate = discriminator(interpolate)`
        4. :math:`loss = \lambda loss\_function(interpolate, d\_interpolate)`
        5. Backpropagate by computing :math:`\nabla loss`
        6. Run a step of the optimizer for discriminator

        Args:
            generator (torchgan.models.Generator): The model to be optimized.
            discriminator (torchgan.models.Discriminator): The discriminator which judges the
                performance of the generator.
            optimizer_discriminator (torch.optim.Optimizer): Optimizer which updates the ``parameters``
                of the ``discriminator``.
            real_inputs (torch.Tensor): The real data to be fed to the ``discriminator``.
            device (torch.device): Device on which the ``generator`` and ``discriminator`` is present.
            batch_size (int): Batch Size of the data infered from the ``DataLoader`` by the ``Trainer``.
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
            if generator.label_type == "none":
                fake = generator(noise)
            elif generator.label_type == "required":
                fake = generator(noise, labels)
            else:
                fake = generator(noise, label_gen)
            eps = torch.rand(1).item()
            interpolate = eps * real_inputs + (1 - eps) * fake
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
