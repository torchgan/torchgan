import torch
import torch.autograd as autograd
import torch.nn.functional as F

from ..utils import reduce

__all__ = [
    "minimax_generator_loss",
    "minimax_discriminator_loss",
    "least_squares_generator_loss",
    "least_squares_discriminator_loss",
    "mutual_information_penalty",
    "wasserstein_generator_loss",
    "wasserstein_discriminator_loss",
    "wasserstein_gradient_penalty",
    "dragan_gradient_penalty",
    "auxiliary_classification_loss",
    "energy_based_generator_loss",
    "energy_based_discriminator_loss",
    "energy_based_pulling_away_term",
    "boundary_equilibrium_generator_loss",
    "boundary_equilibrium_discriminator_loss",
]

# Minimax Losses


def minimax_generator_loss(dgz, nonsaturating=True, reduction="mean"):
    if nonsaturating:
        target = torch.ones_like(dgz)
        return F.binary_cross_entropy_with_logits(dgz, target, reduction=reduction)
    else:
        target = torch.zeros_like(dgz)
        return -1.0 * F.binary_cross_entropy_with_logits(
            dgz, target, reduction=reduction
        )


def minimax_discriminator_loss(dx, dgz, label_smoothing=0.0, reduction="mean"):
    target_ones = torch.ones_like(dgz) * (1.0 - label_smoothing)
    target_zeros = torch.zeros_like(dx)
    loss = F.binary_cross_entropy_with_logits(dx, target_ones, reduction=reduction)
    loss += F.binary_cross_entropy_with_logits(dgz, target_zeros, reduction=reduction)
    return loss


# Least Squared Losses


def least_squares_generator_loss(dgz, c=1.0, reduction="mean"):
    return 0.5 * reduce((dgz - c) ** 2, reduction)


def least_squares_discriminator_loss(dx, dgz, a=0.0, b=1.0, reduction="mean"):
    return 0.5 * (reduce((dx - b) ** 2, reduction) + reduce((dgz - a) ** 2, reduction))


# Mutual Information Penalty


def mutual_information_penalty(c_dis, c_cont, dist_dis, dist_cont, reduction="mean"):
    log_probs = torch.Tensor(
        [
            torch.mean(dist.log_prob(c))
            for dist, c in zip((dist_dis, dist_cont), (c_dis, c_cont))
        ]
    )
    return reduce(-1.0 * log_probs, reduction)


# Wasserstein Losses


def wasserstein_generator_loss(fgz, reduction="mean"):
    return reduce(-1.0 * fgz, reduction)


def wasserstein_discriminator_loss(fx, fgz, reduction="mean"):
    return reduce(fgz - fx, reduction)


def wasserstein_gradient_penalty(interpolate, d_interpolate, reduction="mean"):
    grad_outputs = torch.ones_like(d_interpolate)
    gradients = autograd.grad(
        outputs=d_interpolate,
        inputs=interpolate,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradient_penalty = (gradients.norm(2) - 1) ** 2
    return reduce(gradient_penalty, reduction)


# Dragan Penalty


def dragan_gradient_penalty(interpolate, d_interpolate, k=1.0, reduction="mean"):
    grad_outputs = torch.ones_like(d_interpolate)
    gradients = autograd.grad(
        outputs=d_interpolate,
        inputs=interpolate,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True,
    )[0]

    gradient_penalty = (gradients.norm(2) - k) ** 2
    return reduce(gradient_penalty, reduction)


# Auxiliary Classifier Loss


def auxiliary_classification_loss(logits, labels, reduction="mean"):
    return F.cross_entropy(logits, labels, reduction=reduction)


# Energy Based Losses


def energy_based_generator_loss(dgz, reduction="mean"):
    return reduce(dgz, reduction)


def energy_based_discriminator_loss(dx, dgz, margin, reduction="mean"):
    return reduce(dx + F.relu(-dgz + margin), reduction)


def energy_based_pulling_away_term(d_hid):
    d_hid_normalized = F.normalize(d_hid, p=2, dim=0)
    n = d_hid_normalized.size(0)
    d_hid_normalized = d_hid_normalized.view(n, -1)
    similarity = torch.matmul(d_hid_normalized, d_hid_normalized.transpose(1, 0))
    loss_pt = torch.sum(similarity ** 2) / (n * (n - 1))
    return loss_pt


# Boundary Equilibrium Losses


def boundary_equilibrium_generator_loss(dgz, reduction="mean"):
    return reduce(dgz, reduction)


def boundary_equilibrium_discriminator_loss(dx, dgz, k, reduction="mean"):
    # NOTE(avik-pal): This is a bit peculiar compared to the other losses as it must return 3 values.
    loss_real = reduce(dx, reduction)
    loss_fake = reduce(dgz, reduction)
    loss_total = loss_real - k * loss_fake
    return loss_total, loss_real, loss_fake
