import torch

from ..utils import reduce
from .loss import DiscriminatorLoss, GeneratorLoss

__all__ = ["HistoricalAverageGeneratorLoss", "HistoricalAverageDiscriminatorLoss"]


class HistoricalAverageGeneratorLoss(GeneratorLoss):
    r"""Historical Average Generator Loss from
    `"Improved Techniques for Training GANs
    by Salimans et. al." <https://arxiv.org/pdf/1606.03498.pdf>`_ paper

    The loss can be described as

    .. math:: || \vtheta - \frac{1}{t} \sum_{i=1}^t \vtheta[i] ||^2

    where

    - :math:`G` : Generator
    - :math: `\vtheta[i]` : Generator Parameters at Past Timestep :math: `i`

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output.
            If ``none`` no reduction will be applied. If ``mean`` the outputs are averaged over batch size.
            If ``sum`` the elements of the output are summed.
        override_train_ops (function, optional): Function to be used in place of the default ``train_ops``
        lambd (float, optional): Hyperparameter lambda for scaling the Historical Average Penalty
    """

    def __init__(
        self, reduction="elementwise_mean", override_train_ops=None, lambd=1.0
    ):
        super(HistoricalAverageGeneratorLoss, self).__init__(
            reduction, override_train_ops
        )
        self.timesteps = 0
        self.sum_parameters = []
        self.lambd = lambd

    r"""Defines the standard ``train_ops`` used by historical averaging loss.

        The ``standard optimization algorithm`` for the ``discriminator`` defined in this train_ops
        is as follows:

        1. Compute the loss :math: `|| \vtheta - \frac{1}{t} \sum_{i=1}^t \vtheta[i] ||^2`
        2. Backpropagate by computing :math:`\nabla loss`
        3. Run a step of the optimizer for discriminator

        Args:
            generator (torchgan.models.Generator): The model to be optimized.
            optimizer_generator (torch.optim.Optimizer): Optimizer which updates the ``parameters``
                of the ``generator``.

        Returns:
            Scalar value of the loss.
    """

    def train_ops(self, generator, optimizer_generator):
        if self.override_train_ops is not None:
            return self.override_train_ops(self, generator, optimizer_generator)
        else:
            if self.timesteps == 0:
                for p in generator.parameters():
                    param = p.data.clone()
                    self.sum_parameters.append(param)
                self.timesteps += 1
                return 0.0
            else:
                optimizer_generator.zero_grad()
                loss = 0.0
                for i, p in enumerate(generator.parameters()):
                    loss += torch.sum(
                        (p - (self.sum_parameters[i].data / self.timesteps)) ** 2
                    )
                    self.sum_parameters[i] += p.data.clone()
                self.timesteps += 1
                loss *= self.lambd
                loss.backward()
                optimizer_generator.step()
                return loss.item()


class HistoricalAverageDiscriminatorLoss(DiscriminatorLoss):
    r"""Historical Average Discriminator Loss from
    `"Improved Techniques for Training GANs
    by Salimans et. al." <https://arxiv.org/pdf/1606.03498.pdf>`_ paper

    The loss can be described as

    .. math:: || \vtheta - \frac{1}{t} \sum_{i=1}^t \vtheta[i] ||^2

    where

    - :math:`G` : Discriminator
    - :math: `\vtheta[i]` : Discriminator Parameters at Past Timestep :math: `i`

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output.
            If ``none`` no reduction will be applied. If ``mean`` the outputs are averaged over batch size.
            If ``sum`` the elements of the output are summed.
        override_train_ops (function, optional): Function to be used in place of the default ``train_ops``
        lambd (float, optional): Hyperparameter lambda for scaling the Historical Average Penalty
    """

    def __init__(
        self, reduction="elementwise_mean", override_train_ops=None, lambd=1.0
    ):
        super(HistoricalAverageDiscriminatorLoss, self).__init__(
            reduction, override_train_ops
        )
        self.timesteps = 0
        self.sum_parameters = []
        self.lambd = lambd

    r"""Defines the standard ``train_ops`` used by historical averaging loss.

        The ``standard optimization algorithm`` for the ``discriminator`` defined in this train_ops
        is as follows:

        1. Compute the loss :math: `|| \vtheta - \frac{1}{t} \sum_{i=1}^t \vtheta[i] ||^2`
        2. Backpropagate by computing :math:`\nabla loss`
        3. Run a step of the optimizer for discriminator

        Args:
            generator (torchgan.models.Generator): The model to be optimized.
            optimizer_generator (torch.optim.Optimizer): Optimizer which updates the ``parameters``
                of the ``generator``.

        Returns:
            Scalar value of the loss.
    """

    def train_ops(self, discriminator, optimizer_discriminator):
        if self.override_train_ops is not None:
            return self.override_train_ops(self, discriminator, optimizer_discriminator)
        else:
            if self.timesteps == 0:
                for p in discriminator.parameters():
                    param = p.data.clone()
                    self.sum_parameters.append(param)
                self.timesteps += 1
                return 0.0
            else:
                optimizer_discriminator.zero_grad()
                loss = 0.0
                for i, p in enumerate(discriminator.parameters()):
                    loss += torch.sum(
                        (p - (self.sum_parameters[i].data / self.timesteps)) ** 2
                    )
                    self.sum_parameters[i] += p.data.clone()
                self.timesteps += 1
                loss *= self.lambd
                loss.backward()
                optimizer_discriminator.step()
                return loss.item()
