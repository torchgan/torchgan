import torch
from .loss import GeneratorLoss, DiscriminatorLoss
from ..utils import reduce

__all__ = ['HistoricalAverageGeneratorLoss', 'HistoricalAverageDiscriminatorLoss']

class HistoricalAverageGeneratorLoss(GeneratorLoss):
    # TODO(Aniket1998): Find a generalisation for multi agent GANs
    def __init__(self, reduction='elementwise_mean', override_train_ops=None):
        super(HistoricalAverageGeneratorLoss, self).__init__(reduction, override_train_ops)
        self.timesteps = 0
        self.sum_parameters = []

    def train_ops(self, generator, optimizer_generator):
        if self.override_train_ops is not None:
            return self.override_train_ops(self, generator, optimizer_generator)
        else:
            if self.timesteps == 0:
                for p in generator.parameters():
                    self.sum_parameters.append(torch.Tensor(p))
                self.timesteps += 1
                return 0.0
            else:
                optimizer_generator.zero_grad()
                loss = None
                for i, p in enumerate(generator.parameters()):
                    if loss is None:
                        loss = (p - self.sum_parameters[i]).norm(2)
                    else:
                        loss += (p - self.sum_parameters[i]).norm(2)
                    self.sum_parameters[i] += p
                loss /= self.timesteps
                self.timesteps += 1
                loss.backward()
                optimizer_generator.step()
                return loss.item()

class HistoricalAverageDiscriminatorLoss(DiscriminatorLoss):
    # TODO(Aniket1998): Find a generalisation for multi agent GANs
    def __init__(self, reduction='elementwise_mean', override_train_ops=None):
        super(HistoricalAverageDiscriminatorLoss, self).__init__(reduction, override_train_ops)
        self.timesteps = 0
        self.sum_parameters = []

    def train_ops(self, discriminator, optimizer_discriminator):
        if self.override_train_ops is not None:
            return self.override_train_ops(self, discriminator, optimizer_discriminator)
        else:
            if self.timesteps == 0:
                for p in discriminator.parameters():
                    self.sum_parameters.append(torch.Tensor(p))
                self.timesteps += 1
                return 0.0
            else:
                optimizer_discriminator.zero_grad()
                loss = None
                for i, p in enumerate(discriminator.parameters()):
                    if loss is None:
                        loss = (p - self.sum_parameters[i]).norm(2)
                    else:
                        loss += (p - self.sum_parameters[i]).norm(2)
                    self.sum_parameters[i] += p
                loss /= self.timesteps
                self.timesteps += 1
                loss.backward()
                optimizer_discriminator.step()
                return loss.item()
