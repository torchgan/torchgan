================
torchgan.trainer
================

.. currentmodule:: torchgan.trainer

This subpackage provides ability to perform end to end training capabilities of
the Generator and Discriminator models. It provides strong visualization
capabilities using `tensorboardX <https://github.com/lanpa/tensorboardX>`_. Most of the cases
can be handled elegantly with the default trainer itself. But if incase you need to
`subclass` the trainer for any reason follow the docs closely.

Base Trainer
============

.. autoclass:: BaseTrainer
    :members:

Trainer
=======

.. autoclass:: Trainer
    :members:

Parallel Trainer
================

.. autoclass:: ParallelTrainer
    :members:
