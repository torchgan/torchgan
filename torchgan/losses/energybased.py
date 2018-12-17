import torch
import torch.nn.functional as F
from .loss import GeneratorLoss, DiscriminatorLoss
from ..models import AutoEncodingDiscriminator
from ..utils import reduce

__all__ = ['energy_based_generator_loss', 'energy_based_discriminator_loss',
           'energy_based_pulling_away_term', 'EnergyBasedGeneratorLoss',
           'EnergyBasedDiscriminatorLoss', 'EnergyBasedPullingAwayTerm']

def energy_based_generator_loss(dgz, reduction='elementwise_mean'):
    return reduce(dgz, reduction)

def energy_based_discriminator_loss(dx, dgz, margin, reduction='mean'):
    return reduce(dx + F.relu(-dgz + margin), reduction)

def energy_based_pulling_away_term(d_hid):
    d_hid_normalized = F.normalize(d_hid, p=2, dim=0)
    n = d_hid_normalized.size(0)
    d_hid_normalized = d_hid_normalized.view(n, -1)
    similarity = torch.matmul(d_hid_normalized, d_hid_normalized.transpose(1, 0))
    loss_pt = torch.sum(similarity ** 2) / (n * (n - 1))
    return loss_pt

class EnergyBasedGeneratorLoss(GeneratorLoss):
    r"""Energy Based GAN generator loss from `"Energy Based Generative Adversarial Network
    by Zhao et. al." <https://arxiv.org/abs/1609.03126>`_ paper.

    The loss can be described as:

    .. math:: L(G) = D(G(z))

    where

    - :math:`G` : Generator
    - :math:`D` : Discriminator
    - :math:`z` : A sample from the noise prior

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output.
            If ``none`` no reduction will be applied. If ``mean`` the mean of the output.
            If ``sum`` the elements of the output will be summed.
        override_train_ops (function, optional): A function is passed to this argument,
            if the default ``train_ops`` is not to be used.
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
        return energy_based_generator_loss(dgz, self.reduction)

    def train_ops(self, generator, discriminator, optimizer_generator, device, batch_size, labels=None):
        if isinstance(discriminator, AutoEncodingDiscriminator):
            setattr(discriminator, "embeddings", False)
        loss = super(EnergyBasedGeneratorLoss, self).train_ops(generator, discriminator,
            optimizer_generator, device, batch_size, labels)
        if isinstance(discriminator, AutoEncodingDiscriminator):
            setattr(discriminator, "embeddings", True)
        return loss

class EnergyBasedPullingAwayTerm(GeneratorLoss):
    def __init__(self, reduction='elementwise_mean', pt_ratio=0.1, override_train_ops=None):
        super(EnergyBasedPullingAwayTerm, self).__init__(reduction, override_train_ops)
        self.pt_ratio = pt_ratio

    def forward(self, dgz, d_hid):
        return self.pt_ratio * energy_based_pulling_away_term(d_hid)

    def train_ops(self, generator, discriminator, optimizer_generator, device, batch_size, labels=None):
        if self.override_train_ops is not None:
            return self.override_train_ops(generator, discriminator, optimizer_generator, device, batch_size, labels)
        else:
            if not isinstance(discriminator, AutoEncodingDiscriminator):
                raise Exception('EBGAN PT requires the Discriminator to be a AutoEncoder')
            if not generator.label_type == 'none':
                raise Exception('EBGAN PT supports models which donot require labels')
            if not discriminator.embeddings:
                raise Exception('EBGAN PT requires the embeddings for loss computation')
            noise = torch.randn(batch_size, generator.encoding_dims, device=device)
            optimizer_generator.zero_grad()
            fake = generator(noise)
            d_hid, dgz = discriminator(fake)
            loss = self.forward(dgz, d_hid)
            loss.backward()
            optimizer_generator.step()
            return loss.item()

class EnergyBasedDiscriminatorLoss(DiscriminatorLoss):
    r"""Energy Based GAN generator loss from `"Energy Based Generative Adversarial Network
    by Zhao et. al." <https://arxiv.org/abs/1609.03126>`_ paper

    The loss can be described as:

    .. math:: L(D) = D(x) + max(0, m - D(G(z)))

    where

    - :math:`G` : Generator
    - :math:`D` : Discriminator
    - :math:`m` : Margin Hyperparameter
    - :math:`z` : A sample from the noise prior

    .. note::
        The convergence of EBGAN is highly sensitive to hyperparameters. The ``margin``
        hyperparameter as per the paper was taken as follows:

        +----------------------+--------+
        | Dataset              | Margin |
        +======================+========+
        | MNIST                | 10.0   |
        +----------------------+--------+
        | LSUN                 | 80.0   |
        +----------------------+--------+
        | CELEB A              | 20.0   |
        +----------------------+--------+
        | Imagenet (128 x 128) | 40.0   |
        +----------------------+--------+
        | Imagenet (256 x 256) | 80.0   |
        +----------------------+--------+

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output.
            If ``none`` no reduction will be applied. If ``mean`` the mean of the output.
            If ``sum`` the elements of the output will be summed.
        margin (float, optional): The margin hyperparameter.
        override_train_ops (function, optional): A function is passed to this argument,
            if the default ``train_ops`` is not to be used.
    """
    def __init__(self, reduction='mean', margin=80.0, override_train_ops=None):
        super(EnergyBasedDiscriminatorLoss, self).__init__(reduction, override_train_ops)
        self.margin = margin

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
            scalar if reduction is applied else Tensor with dimensions (N, \*).
        """
        return reduce(dx + F.relu(-dgz + self.margin), self.reduction)

    def train_ops(self, generator, discriminator, optimizer_generator, real_inputs, device,
                  batch_size, labels=None):
        if isinstance(discriminator, AutoEncodingDiscriminator):
            setattr(discriminator, "embeddings", False)
        loss = super(EnergyBasedDiscriminatorLoss, self).train_ops(generator, discriminator,
            optimizer_generator, real_inputs, device, labels)
        if isinstance(discriminator, AutoEncodingDiscriminator):
            setattr(discriminator, "embeddings", True)
        return loss
