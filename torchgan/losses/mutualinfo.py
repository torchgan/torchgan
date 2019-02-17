import torch

from .functional import mutual_information_penalty
from .loss import DiscriminatorLoss, GeneratorLoss

__all__ = ["MutualInformationPenalty"]


class MutualInformationPenalty(GeneratorLoss, DiscriminatorLoss):
    r"""Mutual Information Penalty as defined in
    `"InfoGAN : Interpretable Representation Learning by Information Maximising Generative Adversarial Nets
    by Chen et. al." <https://arxiv.org/abs/1606.03657>`_ paper

    The loss is the variational lower bound of the mutual information between
    the latent codes and the generator distribution and is defined as

    .. math:: L(G,Q) = log(Q|x)

    where

    - :math:`x` is drawn from the generator distribution G(z,c)
    - :math:`c` drawn from the latent code prior :math:`P(c)`

    Args:
        lambd (float, optional): The scaling factor for the loss.
        reduction (str, optional): Specifies the reduction to apply to the output.
            If ``none`` no reduction will be applied. If ``mean`` the mean of the output.
            If ``sum`` the elements of the output will be summed.
        override_train_ops (function, optional): A function is passed to this argument,
            if the default ``train_ops`` is not to be used.
    """

    def __init__(self, lambd=1.0, reduction="mean", override_train_ops=None):
        super(MutualInformationPenalty, self).__init__(reduction, override_train_ops)
        self.lambd = lambd

    def forward(self, c_dis, c_cont, dist_dis, dist_cont):
        r"""Computes the loss for the given input.

        Args:
            c_dis (int): The discrete latent code sampled from the prior.
            c_cont (int): The continuous latent code sampled from the prior.
            dist_dis (torch.distributions.Distribution): The auxilliary distribution :math:`Q(c|x)` over the
                discrete latent code output by the discriminator.
            dist_cont (torch.distributions.Distribution): The auxilliary distribution :math:`Q(c|x)` over the
                continuous latent code output by the discriminator.

        Returns:
            scalar if reduction is applied else Tensor with dimensions (N, \*).
        """
        return mutual_information_penalty(
            c_dis, c_cont, dist_dis, dist_cont, reduction=self.reduction
        )

    def train_ops(
        self,
        generator,
        discriminator,
        optimizer_generator,
        optimizer_discriminator,
        dis_code,
        cont_code,
        device,
        batch_size,
    ):
        if self.override_train_ops is not None:
            self.override_train_ops(
                generator,
                discriminator,
                optimizer_generator,
                optimizer_discriminator,
                dis_code,
                cont_code,
                device,
                batch_size,
            )
        else:
            noise = torch.randn(batch_size, generator.encoding_dims, device=device)
            optimizer_discriminator.zero_grad()
            optimizer_generator.zero_grad()
            fake = generator(noise, dis_code, cont_code)
            _, dist_dis, dist_cont = discriminator(fake, True)
            loss = self.forward(dis_code, cont_code, dist_dis, dist_cont)
            weighted_loss = self.lambd * loss
            weighted_loss.backward()
            optimizer_discriminator.step()
            optimizer_generator.step()
            return weighted_loss.item()
