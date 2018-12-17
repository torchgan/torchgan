import torch
from .loss import GeneratorLoss, DiscriminatorLoss
from ..utils import reduce

__all__ = ['HistoricalAverageGeneratorLoss', 'HistoricalAverageDiscriminatorLoss']

class HistoricalAverageGeneratorLoss(GeneratorLoss):
    # TODO(Aniket1998): Find a generalisation for multi agent GANs
    def __init__(self, reduction='elementwise_mean', override_train_ops=None, lambd=1.0):
        super(HistoricalAverageGeneratorLoss, self).__init__(reduction, override_train_ops)
        self.timesteps = 0
        self.sum_parameters = []
        self.lambd = lambd

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
                    loss += torch.sum((p - (self.sum_parameters[i].data / self.timesteps)) ** 2)
                    self.sum_parameters[i] += p.data.clone()
                self.timesteps += 1
                loss *= self.lambd
                loss.backward()
                optimizer_generator.step()
                return loss.item()

class HistoricalAverageDiscriminatorLoss(DiscriminatorLoss):
    # TODO(Aniket1998): Find a generalisation for multi agent GANs
    def __init__(self, reduction='elementwise_mean', override_train_ops=None, lambd=1.0):
        super(HistoricalAverageDiscriminatorLoss, self).__init__(reduction, override_train_ops)
        self.timesteps = 0
        self.sum_parameters = []
        self.lambd = lambd

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
                    loss += torch.sum((p - (self.sum_parameters[i].data / self.timesteps)) ** 2)
                    self.sum_parameters[i] += p.data.clone()
                self.timesteps += 1
                loss *= self.lambd
                loss.backward()
                optimizer_discriminator.step()
                return loss.item()
