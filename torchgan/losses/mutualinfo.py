import torch
from .loss import GeneratorLoss, DiscriminatorLoss

__all__ = ['MutualInformationPenalty']

class MutualInformationPenalty(GeneratorLoss, DiscriminatorLoss):
    """Mutual Information Penalty as defined in
    "InfoGAN : Interpretable Representation Learning by Information Maximising Generative Adversarial Nets
    by Chen et. al." <https://arxiv.org/abs/1606.03657>

    The loss is the variational lower bound of the mutual information between
    the latent codes and the generator distribution and is defined as

    L(G,Q) = log(Q|x)
    where x is drawn from the generator distribution G(z,c)
    c drawn from the latent code prior P(c)

    Args:
        c_dis(Tensor) : The discrete latent code sampled from the prior
        c_cont(Tensor) : The continuous latent code sampled from the prior
        dist_dis : The auxilliary distribution Q(c|x) over the discrete latent code output by the discriminator
        dist_cont : The auxilliary distribution Q(c|x) over the continuous latent code output by the discriminator
        """
    def forward(self, c_dis, c_cont, dist_dis, dist_cont):
        log_prob_dis = dist_dis.log_prob(c_dis)
        log_prob_cont = dist_cont.log_prob(c_cont)

        if self.reduction == 'elementwise_mean':
            return -1.0 * torch.mean(log_prob_dis + log_prob_cont)
        elif self.reduction == 'sum':
            return -1.0 * torch.sum(log_prob_dis + log_prob_cont)
        else:
            return -(log_prob_dis + log_prob_cont)
