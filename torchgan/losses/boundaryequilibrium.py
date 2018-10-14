import torch
from .loss import GeneratorLoss, DiscriminatorLoss
from ..utils import reduce

__all__ = ['BoundaryEquilibriumGeneratorLoss', 'BoundaryEquilibriumDiscriminatorLoss']

class BoundaryEquilibriumGeneratorLoss(GeneratorLoss):
    def forward(self, dgz):
        return reduce(dgz, self.reduction)

class BoundaryEquilibriumDiscriminatorLoss(DiscriminatorLoss):
    def __init__(self, reduction='elementwise_mean', override_train_ops=None, init_k=0.0, lambd=0.001, gamma=0.75):
        self.reduction = reduction
        self.override_train_ops = override_train_ops
        self.k = init_k
        self.lambd = lambd
        self.gamma = gamma
        self.convergence_metric = None

    def forward(self, dx, dgz):
        loss_real = reduce(dx, self.reduction)
        loss_fake = reduce(dgz, self.reduction)
        loss_total = loss_real - self.k * loss_fake
        return loss_total, loss_real, loss_fake

    def set_k(self, k=0.0):
        self.k = k

    def update_k(self, loss_real, loss_fake):
        diff = self.gamma * loss_real - loss_fake
        self.k += self.lambd * diff
        self.convergence_metric = loss_real + abs(diff)
        if self.k < 0.0:
            self.k = 0.0
        elif self.k > 1.0:
            self.k = 1.0

    def train_ops(self, generator, discriminator, optimizer_discriminator, real_inputs, noise, labels_provided=False):
        if self.override_train_ops is not None:
            return self.override_train_ops(generator, discriminator, optimizer_discriminator,
                                           real_inputs, noise, labels_provided)
        else:
            real = real_inputs if labels_provided is False else real_inputs[0]
            optimizer_discriminator.zero_grad()
            dx = discriminator(real)
            fake = generator(noise)
            dgz = discriminator(fake.detach())
            loss_total, loss_real, loss_fake = self.forward(dx, dgz)
            loss_total.backward()
            optimizer_discriminator.step()
            self.update_k(loss_real.item(), loss_fake.item())
            return loss_total.item()
