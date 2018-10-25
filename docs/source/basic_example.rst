Starter Example
===============

As a starter example we will try to train a `DCGAN` on `CIFAR-10`. `DCGAN` is in-built into to the library, but let it not fool you into believing that we can only use this package for some fixed limited tasks. This library is fully customizable. For that have a look at the `Examples`.

But for now let us just use this as a small demo example

First we import the necessary files

.. code:: python

    import torch
    import torchvision
    from torch.optim import Adam
    import torch.utils.data as data
    import torchvision.datasets as dsets
    import torchvision.transforms as transforms
    from torchgan import *
    from torchgan.models import SmallDCGANGenerator, SmallDCGANDiscriminator
    from torchgan.losses import MinimaxGeneratorLoss, MinimaxDiscriminatorLoss,
    from torchgan.trainer import Trainer

Now write a function which returns the `data loader` for `CIFAR10`.

.. code:: python

    def cifar10_dataloader():
        train_dataset = dsets.CIFAR10(root='/data/avikpal', train=True,
                                      transform=transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))]),
                                      download=True)
        train_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)
        return train_loader

Now lets us create the `Trainer` object and pass the `data loader` to it.

.. code:: python

    trainer = Trainer({"generator": {"name": DCGANGenerator, "args": {"out_channels": 3, "step_channels": 16}},
                       "discriminator": {"name": DCGANDiscriminator, "args": {"in_channels": 3, "step_channels": 16}}},
                      {"optimizer_generator": {"name": Adam, "args": {"lr": 0.0002, "betas": (0.5, 0.999)}},
                       "optimizer_discriminator": {"name": Adam, "args": {"lr": 0.0002, "betas": (0.5, 0.999)}}},
                      [MinimaxGeneratorLoss(), MinimaxDiscriminatorLoss()],
                      sample_size=64, epochs=20)

    trainer(cifar10_dataloader())

Now log into `tensorboard` and visualize the training process.
