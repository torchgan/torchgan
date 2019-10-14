import os
import sys
import unittest

import torch
import torch.utils.data as data
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.optim import Adam
from torchgan import *
from torchgan.losses import *
from torchgan.metrics import *
from torchgan.models import *
from torchgan.trainer import Trainer

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


def mnist_dataloader():
    train_dataset = dsets.MNIST(
        root="./mnist",
        train=True,
        transform=transforms.Compose(
            [
                transforms.Pad((2, 2)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),
            ]
        ),
        download=True,
    )
    train_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    return train_loader


class TestTrainer(unittest.TestCase):
    def test_trainer_dcgan(self):
        network_params = {
            "generator": {
                "name": DCGANGenerator,
                "args": {"out_channels": 1, "step_channels": 4},
                "optimizer": {
                    "name": Adam,
                    "args": {"lr": 0.0002, "betas": (0.5, 0.999)},
                },
            },
            "discriminator": {
                "name": DCGANDiscriminator,
                "args": {"in_channels": 1, "step_channels": 4},
                "optimizer": {
                    "name": Adam,
                    "args": {"lr": 0.0002, "betas": (0.5, 0.999)},
                },
            },
        }
        losses_list = [MinimaxGeneratorLoss(), MinimaxDiscriminatorLoss()]
        trainer = Trainer(
            network_params,
            losses_list,
            sample_size=1,
            epochs=1,
            device=torch.device("cpu"),
        )
        trainer(mnist_dataloader())

    def test_trainer_cgan(self):
        network_params = {
            "generator": {
                "name": ConditionalGANGenerator,
                "args": {"num_classes": 10, "out_channels": 1, "step_channels": 4},
                "optimizer": {
                    "name": Adam,
                    "args": {"lr": 0.0002, "betas": (0.5, 0.999)},
                },
            },
            "discriminator": {
                "name": ConditionalGANDiscriminator,
                "args": {"num_classes": 10, "in_channels": 1, "step_channels": 4},
                "optimizer": {
                    "name": Adam,
                    "args": {"lr": 0.0002, "betas": (0.5, 0.999)},
                },
            },
        }
        losses_list = [MinimaxGeneratorLoss(), MinimaxDiscriminatorLoss()]
        trainer = Trainer(
            network_params,
            losses_list,
            sample_size=1,
            epochs=1,
            device=torch.device("cpu"),
        )
        trainer(mnist_dataloader())

    def test_trainer_acgan(self):
        network_params = {
            "generator": {
                "name": ACGANGenerator,
                "args": {"num_classes": 10, "out_channels": 1, "step_channels": 4},
                "optimizer": {
                    "name": Adam,
                    "args": {"lr": 0.0002, "betas": (0.5, 0.999)},
                },
            },
            "discriminator": {
                "name": ACGANDiscriminator,
                "args": {"num_classes": 10, "in_channels": 1, "step_channels": 4},
                "optimizer": {
                    "name": Adam,
                    "args": {"lr": 0.0002, "betas": (0.5, 0.999)},
                },
            },
        }
        losses_list = [
            MinimaxGeneratorLoss(),
            MinimaxDiscriminatorLoss(),
            AuxiliaryClassifierGeneratorLoss(),
            AuxiliaryClassifierDiscriminatorLoss(),
        ]
        trainer = Trainer(
            network_params,
            losses_list,
            sample_size=1,
            epochs=1,
            device=torch.device("cpu"),
        )
        trainer(mnist_dataloader())
