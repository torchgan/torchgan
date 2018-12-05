import torch
import torch.nn.functional as F
from .loss import GeneratorLoss, DiscriminatorLoss
from ..utils import reduce

__all__ = ['FeatureMatchingGeneratorLoss']

class FeatureMatchingGeneratorLoss(GeneratorLoss):
    r"""Feature Matching Generator loss from
    `"Improved Training of GANs by Salimans et. al." <https://arxiv.org/abs/1701.07875>`_ paper

    The loss can be described as:

    .. math:: L(G) = ||f(x)-f(G(z))||

    where

    - G : Generator
    - f : An intermediate activation from the discriminator
    - z : A sample from the noise prior

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output.
            If `none` no reduction will be applied. If `elementwise_mean` the sum of
            the elements will be divided by the number of elements in the output. If
            `sum` the output will be summed.
    """
    def forward(self, fx, fgz):
        return F.mse_loss(fgz, fx, reduction=self.reduction)

    def train_ops(self, generator, discriminator, optimizer_generator, real_inputs, device, batch_size, labels=None):
        if self.override_train_ops is not None:
            return self.override_train_ops(generator, discriminator, optimizer_generator, device, batch_size, labels)
        else:
            if labels is None and generator.label_type == 'required':
                raise Exception('GAN model requires labels for training')
            noise = torch.randn(real_inputs.size(0), generator.encoding_dims, device=device)
            optimizer_generator.zero_grad()
            if generator.label_type == 'generated':
                label_gen = torch.randint(0, generator.num_classes, (real_inputs.size(0),), device=device)
            if generator.label_type == 'none':
                fake = generator(noise)
            elif generator.label_type == 'required':
                fake = generator(noise, labels)
            elif generator.label_type == 'generated':
                fake = generator(noise, label_gen)

            if discriminator.label_type == 'none':
                fx = discriminator(real_inputs, feature_matching=True)
                fgz = discriminator(fake, feature_matching=True)
            else:
                if discriminator.label_type == 'generated':
                    fx = discriminator(real_inputs, label_gen, feature_matching=True)
                else:
                    fx = discriminator(real_inputs, labels, feature_matching=True)
                if generator.label_type == 'generated':
                    fgz = discriminator(fake, label_gen, feature_matching=True)
                else:
                    fgz = discriminator(fake, labels, feature_matching=True)
            loss = self.forward(fx, fgz)
            loss.backward()
            optimizer_generator.step()
            return loss.item()
