import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss import GeneratorLoss, DiscriminatorLoss
from ..utils import reduce

__all__ = ['AuxiliaryClassifierGeneratorLoss', 'AuxiliaryClassifierDiscriminatorLoss']

class AuxiliaryClassifierGeneratorLoss(GeneratorLoss):
    r"""Auxiliary Classifier GAN (ACGAN) loss based on a from
    `"Conditional Image Synthesis With Auxiliary Classifier GANs
    by Odena et. al. " <https://arxiv.org/abs/1610.09585>`_ paper

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output.
            If `none` no reduction will be applied. If `elementwise_mean` the sum of
            the elements will be divided by the number of elements in the output. If
            `sum` the output will be summed.
    """
    def forward(self, logits, labels):
        return F.cross_entropy(logits, labels, reduction=self.reduction)

    def train_ops(self, generator, discriminator, optimizer_generator, device, batch_size, labels=None):
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
            label_gen = torch.randint(0, generator.num_classes, (batch_size,), device=device)
            fake = generator(noise, label_gen)
        cgz = discriminator(fake, mode='classifier')
        if generator.label_type == 'required':
            loss = self.forward(cgz, labels)
        else:
            label_gen = label_gen.type(torch.LongTensor).to(device)
            loss = self.forward(cgz, label_gen)
        loss.backward()
        optimizer_generator.step()
        return loss.item()

class AuxiliaryClassifierDiscriminatorLoss(DiscriminatorLoss):
    r"""Auxiliary Classifier GAN (ACGAN) loss based on a from
    `"Conditional Image Synthesis With Auxiliary Classifier GANs
    by Odena et. al. " <https://arxiv.org/abs/1610.09585>`_ paper

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output.
            If `none` no reduction will be applied. If `elementwise_mean` the sum of
            the elements will be divided by the number of elements in the output. If
            `sum` the output will be summed.
    """
    def forward(self, logits, labels):
        return F.cross_entropy(logits, labels, reduction=self.reduction)

    def train_ops(self, generator, discriminator, optimizer_discriminator, real_inputs, batch_size,
                  device, labels=None):
        if self.override_train_ops is not None:
            return self.override_train_ops(generator, discriminator, optimizer_discriminator,
                    real_inputs, batch_size, device, labels)
        if labels is None:
            raise Exception('ACGAN Discriminator requires labels for training')
        if generator.label_type is 'none':
            raise Exception('Incorrect Model: ACGAN generator must require labels for training')
        noise = torch.randn(batch_size, generator.encoding_dims, device=device)
        optimizer_discriminator.zero_grad()
        cx = discriminator(real_inputs, mode='classifier')
        if generator.label_type == 'required':
            fake = generator(noise, labels)
        elif generator.label_type == 'generated':
            label_gen = torch.randint(0, generator.num_classes, (batch_size,), device=device)
            fake = generator(noise, label_gen)
        cgz = discriminator(fake, mode='classifier')
        if generator.label_type == 'required':
            loss = self.forward(cgz, labels) + self.forward(cx, labels)
        else:
            label_gen = label_gen.type(torch.LongTensor).to(device)
            loss = self.forward(cgz, label_gen) + self.forward(cx, labels)
        loss.backward()
        optimizer_discriminator.step()
        return loss.item()
