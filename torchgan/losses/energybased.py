import torch
import torch.nn.functional as F
from .loss import GeneratorLoss, DiscriminatorLoss
from ..utils import reduce

__all__ = ['energy_based_generator_loss', 'energy_based_discriminator_loss',
           'energy_based_pulling_away_term', 'EnergyBasedGeneratorLoss',
           'EnergyBasedDiscriminatorLoss', 'EnergyBasedPullingAwayTerm']

def energy_based_generator_loss(dgz, reduction='elementwise_mean'):
    return reduce(dgz, reduction)

def energy_based_discriminator_loss(dx, dgz, margin, reduction='elementwise_mean'):
    return reduce(dx + F.relu(-dgz + margin), reduction)

def energy_based_pulling_away_term(d_hid):
    d_hid_normalized = d_hid / torch.sqrt(torch.sum(d_hid ** 2, -1, keepdim=True))
    similarity = torch.matmul(d_hid_normalized, d_hid_normalized.transpose(1, 0))
    n = d_hid_normalized.size(0)
    loss_pt = torch.sum(similarity ** 2) / (n * (n - 1))
    return loss_pt

class EnergyBasedGeneratorLoss(GeneratorLoss):
    r"""Energy Based GAN generator loss from
    `"Energy Based Generative Adversarial Network
    by Zhao et. al." <https://arxiv.org/abs/1609.03126>`_ paper.

    The loss can be described as:

    .. math:: L(G) = D(G(z))

    where

    - G : Generator
    - D : Discriminator
    - z : A sample from the noise prior

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output.
            If `none` no reduction will be applied. If `elementwise_mean` the sum of
            the elements will be divided by the number of elements in the output. If
            `sum` the output will be summed.
    """
    def __init__(self, reduction='elementwise_mean', override_train_ops=None):
        super(EnergyBasedGeneratorLoss, self).__init__(reduction, override_train_ops)

    def forward(self, dgz):
        r"""
        Args:
            dgz (torch.Tensor) : Output of the Generator. It must have the dimensions
                                 (N, \*) where \* means any number of additional dimensions.

        Returns:
            scalar if reduction is applied else Tensor with dimensions (N, \*).
        """
        return energy_based_generator_loss(dgz, self.reduction)

class EnergyBasedPullingAwayTerm(GeneratorLoss):
    def __init__(self, reduction='elementwise_mean', pt_ratio=0.1, override_train_ops=None):
        super(EnergyBasedPullingAwayTerm, self).__init__(reduction, override_train_ops)
        self.pt_ratio = pt_ratio

    def forward(self, dgz, d_hid):
        return energy_based_generator_loss(dgz, self.reduction) +\
               self.pt_ratio * energy_based_pulling_away_term(d_hid)

    def train_ops(self, generator, discriminator, optimizer_generator, device, batch_size, labels=None):
        if self.override_train_ops is not None:
            return self.override_train_ops(generator, discriminator, optimizer_generator, device, batch_size, labels)
        else:
            if not isinstance(discriminator, 'AutoEncodingDiscriminator'):
                raise Exception('EBGAN PT requires the Discriminator to be a AutoEncoder')
            if not generator.label_type == 'none':
                raise Exception('EBGAN PT supports models which donot require labels')
            if discriminator.embeddings == False:
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
    r"""Energy Based GAN generator loss from
    `"Energy Based Generative Adversarial Network
    by Zhao et. al." <https://arxiv.org/abs/1609.03126>`_ paper

    The loss can be described as:

    .. math:: L(D) = D(x) + max(0, m - D(G(z)))

    where

    - G : Generator
    - D : Discriminator
    - m : Margin Hyperparameter (default 80.0)
    - z : A sample from the noise prior

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output.
            If `none` no reduction will be applied. If `elementwise_mean` the sum of
            the elements will be divided by the number of elements in the output. If
            `sum` the output will be summed.
    """

    def __init__(self, reduction='elementwise_mean', margin=80.0, override_train_ops=None):
        super(EnergyBasedDiscriminatorLoss, self).__init__(reduction, override_train_ops)
        self.margin = margin

    def forward(self, dx, dgz):
        r"""
        Args:
            dx (torch.Tensor) : Output of the Discriminator. It must have the dimensions
                                (N, \*) where \* means any number of additional dimensions.
            dgz (torch.Tensor) : Output of the Generator. It must have the dimensions
                                 (N, \*) where \* means any number of additional dimensions.

        Returns:
            scalar if reduction is applied else Tensor with dimensions (N, \*).
        """
        return reduce(dx + F.relu(-dgz + self.margin), self.reduction)
