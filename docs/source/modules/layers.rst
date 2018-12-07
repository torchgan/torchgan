===============
torchgan.layers
===============

.. currentmodule:: torchgan.layers

This layers subpackage is a collection of popular building blocks
for GAN architectures. Currently the following blocks are supported:

.. contents::
    :local:

ResidualBlock
==============
.. autoclass:: ResidualBlock
    :members:

DensenetBlocks
==============
.. autoclass:: BasicBlock
    :members:
.. autoclass:: BottleneckBlock
    :members:
.. autoclass:: TransitionBlock
    :members:
.. autoclass:: TransitionBlockTranspose
    :members:
.. autoclass:: DenseBlock
    :members:

SelfAttention
=============
.. autoclass:: SelfAttention
    :members:

SpectralNorm2d
==============
.. autoclass:: SpectralNorm2d
    :members:
