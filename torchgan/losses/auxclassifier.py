import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss import GeneratorLoss, DiscriminatorLoss
from ..utils import reduce

__all__ = ['AuxiliaryClassifierGeneratorLoss', 'AuxiliaryClassifierDiscriminatorLoss',
        'CombineAuxiliaryClassifierGeneratorLoss', 'CombineAuxiliaryClassifierDiscriminatorLoss']

class AuxiliaryClassifierGeneratorLoss(GeneratorLoss):
    def forward(self, logits, labels):
        return F.cross_entropy(logits, labels, reduction=self.reduction)

    def train_ops(self, generator, discriminator, optimizer_generator,
            device, batch_size, labels=None):
        if self.override_train_ops is not None:
            return self.override_train_ops(generator, discriminator, optimizer_generator, device, batch_size, labels)
        if generator.label_type == 'required' and labels is None:
            raise Exception('GAN model requires label for training')
        noise = torch.randn(batch_size, generator.encoding_dims, device=device)
        optimizer_generator.zero_grad()
        if generator.label_type == 'none':
            raise Exception('Incorrect Model: ACGAN generator must require labels')
        if generator.label_type == 'required':
            fake = generator(noise, labels)
        elif generator.label_type == 'generated':
            label_gen = torch.randint(0, generator.num_classes, (batch_size,))
            fake = generator(noise, label_gen)
        cgz = discriminator(fake, mode='classifier')
        if generator.label_type == 'required':
            loss = self.forward(cgz, labels)
        else:
            loss = self.forward(cgz, label_gen)
        loss.backward()
        optimizer_generator.step()
        return loss.item()
