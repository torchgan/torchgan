import torch
import torch.nn.functional as F

from ..utils import reduce
from .loss import DiscriminatorLoss, GeneratorLoss

__all__ = ["FeatureMatchingGeneratorLoss"]


class FeatureMatchingGeneratorLoss(GeneratorLoss):
    r"""Feature Matching Generator loss from
    `"Improved Training of GANs by Salimans et. al." <https://arxiv.org/abs/1606.03498>`_ paper

    The loss can be described as:

    .. math:: L(G) = ||f(x)-f(G(z))||_2

    where

    - :math:`G` : Generator
    - :math:`f` : An intermediate activation from the discriminator
    - :math:`z` : A sample from the noise prior

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output.
            If ``none`` no reduction will be applied. If ``mean`` the outputs are averaged over batch size.
            If ``sum`` the elements of the output are summed.
        override_train_ops (function, optional): Function to be used in place of the default ``train_ops``
    """

    def forward(self, fx, fgz):
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
        return F.mse_loss(fgz, fx, reduction=self.reduction)

    def train_ops(
        self,
        generator,
        discriminator,
        optimizer_generator,
        real_inputs,
        device,
        labels=None,
    ):
        r"""Defines the standard ``train_ops`` used for feature matching.

        The ``standard optimization algorithm`` for the ``generator`` defined in this train_ops
        is as follows:

        1. :math:`fake = generator(noise)`
        2. :math:`value_1 = discriminator(fake)` where :math:`value_1` is an activation of an intermediate
                                                 discriminator layer
        3. :math:`value_2 = discriminator(real)` where :math:`value_2` is an activation of the same intermediate
                                                 discriminator layer
        4. :math:`loss = loss\_function(value_1, value_2)`
        5. Backpropagate by computing :math:`\nabla loss`
        6. Run a step of the optimizer for generator

        Args:
            generator (torchgan.models.Generator): The model to be optimized.
            discriminator (torchgan.models.Discriminator): The discriminator which judges the
                performance of the generator.
            optimizer_generator (torch.optim.Optimizer): Optimizer which updates the ``parameters``
                of the ``generator``.
            real_inputs (torch.Tensor): The real data to be fed to the ``discriminator``.
            device (torch.device): Device on which the ``generator`` and ``discriminator`` is present.
            labels (torch.Tensor, optional): Labels for the data.

        Returns:
            Scalar value of the loss.
        """
        if self.override_train_ops is not None:
            return self.override_train_ops(
                generator, discriminator, optimizer_generator, device, labels
            )
        else:
            if labels is None and generator.label_type == "required":
                raise Exception("GAN model requires labels for training")
            batch_size = real_inputs.size(0)
            noise = torch.randn(batch_size, generator.encoding_dims, device=device)
            optimizer_generator.zero_grad()
            if generator.label_type == "generated":
                label_gen = torch.randint(
                    0, generator.num_classes, (batch_size,), device=device
                )
            if generator.label_type == "none":
                fake = generator(noise)
            elif generator.label_type == "required":
                fake = generator(noise, labels)
            elif generator.label_type == "generated":
                fake = generator(noise, label_gen)

            if discriminator.label_type == "none":
                fx = discriminator(real_inputs, feature_matching=True)
                fgz = discriminator(fake, feature_matching=True)
            else:
                if discriminator.label_type == "generated":
                    fx = discriminator(real_inputs, label_gen, feature_matching=True)
                else:
                    fx = discriminator(real_inputs, labels, feature_matching=True)
                if generator.label_type == "generated":
                    fgz = discriminator(fake, label_gen, feature_matching=True)
                else:
                    fgz = discriminator(fake, labels, feature_matching=True)
            loss = self.forward(fx, fgz)
            loss.backward()
            optimizer_generator.step()
            return loss.item()
