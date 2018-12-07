"""
===========================
SELF ATTENTION GAN TUTORIAL
===========================

**Author** : `Avik Pal <https://avik-pal.github.io>`_

INTRODUCTION
============

In this tutorial we shall implement the architecture of a generative adversarial
network as described in the paper `"Self Attention Generative Adversarial Networks by Zhang et. al." <https://arxiv.org/abs/1805.08318>`_
This tutorial demonstrates how to use the ``layers`` API of ``torchgan`` to
design custom ``models``. For this demonstration we shall be using the CIFAR10
dataset. We shall show how to write your own ``Generator`` and ``Discriminator``
and then use the default ``losses`` provided by ``torchgan``.
"""

import os
import random
import matplotlib.pyplot as plt
import numpy as np
# Turn off Visdom Logging
os.environ["VISDOM_LOGGING"] = "0"

##################################################################################

# Pytorch and Torchvision Imports
import torch
import torchvision
from torch.optim import Adam
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as vutils

##################################################################################

# Torchgan Imports
import torchgan
from torchgan.layers import SelfAttention, SpectralNorm2d
from torchgan.models import Generator, Discriminator
from torchgan.losses import WassersteinGeneratorLoss, WassersteinDiscriminatorLoss
from torchgan.trainer import Trainer

##################################################################################

# Set random seem for reproducibility
manualSeed = 42
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

##################################################################################
# DATASET
# =======
# ``torchvision.datasets`` API allows us to directly download and use the CIFAR10
# Dataset. We need to ``Normalize`` the Image Tensors to have a mean of 0.5 and
# standard deviation of 0.5 across all channels.
dataset = dsets.MNIST(root='./mnist', train=True,
                      transform=transforms.Compose([transforms.Pad((2, 2)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))]),
                      download=True)

##################################################################################
# Input to the ``torchgan.trainer.Trainer`` must be a ``torch.data.DataLoader``,
# hence we create a DataLoader for the CIFAR10 dataset.
dataloader = data.DataLoader(dataset, batch_size=64, shuffle=True)

##################################################################################
# LAYERS API
# ==========
# The ``torchgan.layers`` API provides the building blocks for various complex
# models. For a detailed list of all the features provided go to `Layers
# Documentation <https://torchgan.readthedocs.io/en/latest/modules/layers.html>`_
#
# Next we need some explanation about Spectral Norm and Self Attention
#
# After the explanation we show how to create the models
class SAGANGenerator(Generator):
    def __init__(self, encoding_dims=100, step_channels=64):
        super(SAGANGenerator, self).__init__(encoding_dims, 'none')
        d = int(step_channels * 4)
        self.model = nn.Sequential(
            SpectralNorm2d(nn.ConvTranspose2d(self.encoding_dims, d, 4, 1, 0, bias=True)),
            nn.LeakyReLU(0.2),
            SpectralNorm2d(nn.ConvTranspose2d(d, d // 2, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.2),
            SelfAttention(d // 2),
            SpectralNorm2d(nn.ConvTranspose2d(d // 2, d // 4, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.2),
            SelfAttention(d // 4),
            SpectralNorm2d(nn.ConvTranspose2d(d // 4, 1, 4, 2, 1, bias=True)), nn.Tanh())

    def forward(self, x):
        x = x.view(-1, x.size(1), 1, 1)
        return self.model(x)

##################################################################################
# We need another explanation for the Discriminator model

class SAGANDiscriminator(Discriminator):
    def __init__(self, step_channels=64):
        super(SAGANDiscriminator, self).__init__(1, 'none')
        d = step_channels
        self.model = nn.Sequential(
            SpectralNorm2d(nn.Conv2d(self.input_dims, d, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.2),
            SpectralNorm2d(nn.Conv2d(d, d * 2, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.2),
            SelfAttention(d * 2),
            SpectralNorm2d(nn.Conv2d(d * 2, d * 4, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.2),
            SelfAttention(d * 4),
            SpectralNorm2d(nn.Conv2d(d * 4, 1, 4, 1, 0, bias=True)), nn.Tanh())

    def forward(self, x):
        return self.model(x)

##################################################################################
# MODEL HYPERPARAMETERS
# =====================
# Mention about TTUR and stuff

# network_params = {
#     "generator": {"name": SAGANGenerator, "args": {"step_channels": 32},
#                   "optimizer": {"name": Adam, "args": {"lr": 0.0001, "betas": (0.5, 0.999)}}},
#     "discriminator": {"name": SAGANDiscriminator, "args": {"step_channels": 32},
#                       "optimizer": {"name": Adam, "args": {"lr": 0.0002, "betas": (0.5, 0.999)}}}
# }
network_params = {
    "generator": {"name": SAGANGenerator, "args": {"step_channels": 16}},
    "discriminator": {"name": SAGANDiscriminator, "args": {"step_channels": 16}}
}
optim_params = {
    "optimizer_generator": {"name": Adam, "args": {"lr": 0.0001, "betas": (0.5, 0.999)}},
    "optimizer_discriminator": {"name": Adam, "args": {"lr": 0.0002, "betas": (0.5, 0.999)}}
}

##################################################################################
# We train for a longer period if GPU is available

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    epochs = 50
else:
    device = torch.device("cpu")
    epochs = 5

print("Device: {}".format(device))
print("Epochs: {}".format(epochs))

##################################################################################
# Plot some of the training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

##################################################################################

losses_list = [WassersteinGeneratorLoss(), WassersteinDiscriminatorLoss()]

##################################################################################
# TRAINER
# =======

# trainer = Trainer(network_params, losses_list, sample_size=64, epochs=epochs, device=device)
trainer = Trainer(network_params, optim_params, losses_list, sample_size=64, epochs=epochs, device=device)

# trainer(dataloader)

trainer.complete()

##################################################################################
# REAL vs FAKE IMAGES
# ===================

# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(plt.imread("{}/epoch{}_generator.png".format(trainer.recon, epochs)))
plt.show()
