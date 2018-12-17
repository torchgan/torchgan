===============
torchgan.losses
===============

.. currentmodule:: torchgan.losses

This losses subpackage is a collection of popular loss functions used
in the training of GANs. Currently the following losses are supported:

.. contents::
    :local:

These losses are tested with the current available trainers. So if you need
to implement you custom loss for using with the :mod:`trainer` it is recommended
that you subclass the :class:`GeneratorLoss` and :class:`DiscriminatorLoss`.

.. warning::
    The ``override_train_ops`` gets only the arguments that were received by the
    default ``train_ops``. Hence it might not be a wise to use this very often.
    If this is used make sure to take into account the arguments and their order.
    A better alternative is to subclass the Loss and define a custom ``train_ops``.

.. warning::
    ``train_ops`` are designed to be used internally through the ``Trainer``. Hence it
    is highly recommended that this function is not directly used by external sources,
    i.e. no call to this function is made outside the ``Trainer``.

Loss
====

GeneratorLoss
-------------

.. autoclass:: GeneratorLoss
    :members:

DiscriminatorLoss
-----------------

.. autoclass:: DiscriminatorLoss
    :members:

Least Squares Loss
==================

LeastSquaresGeneratorLoss
-------------------------

.. autoclass:: LeastSquaresGeneratorLoss
    :members:

LeastSquaresDiscriminatorLoss
-----------------------------

.. autoclass:: LeastSquaresDiscriminatorLoss
    :members:

Minimax Loss
============

MinimaxGeneratorLoss
--------------------

.. autoclass:: MinimaxGeneratorLoss
    :members:

MinimaxDiscriminatorLoss
------------------------

.. autoclass:: MinimaxDiscriminatorLoss
    :members:

Boundary Equilibrium Loss
=========================

BoundaryEquilibriumGeneratorLoss
--------------------------------

.. autoclass:: BoundaryEquilibriumGeneratorLoss
    :members:

BoundaryEquilibriumDiscriminatorLoss
------------------------------------

.. autoclass:: BoundaryEquilibriumDiscriminatorLoss
    :members:

Energy Based Loss
=================

EnergyBasedGeneratorLoss
------------------------

.. autoclass:: EnergyBasedGeneratorLoss
    :members:

EnergyBasedDiscriminatorLoss
----------------------------

.. autoclass:: EnergyBasedDiscriminatorLoss
    :members:

EnergyBasedPullingAwayTerm
--------------------------

.. autoclass:: EnergyBasedPullingAwayTerm
    :members:

Wasserstein Loss
================

WassersteinGeneratorLoss
------------------------

.. autoclass:: WassersteinGeneratorLoss
    :members:

WassersteinDiscriminatorLoss
----------------------------

.. autoclass:: WassersteinDiscriminatorLoss
    :members:

WassersteinGradientPenalty
--------------------------

.. autoclass:: WassersteinGradientPenalty
    :members:

Mutual Information Penalty
==========================

.. autoclass:: MutualInformationPenalty
    :members:

Dragan Loss
===========

DraganGradientPenalty
---------------------

.. autoclass:: DraganGradientPenalty
    :members:

Auxillary Classifier Loss
=========================

AuxiliaryClassifierGeneratorLoss
--------------------------------

.. autoclass:: AuxiliaryClassifierGeneratorLoss
    :members:

AuxiliaryClassifierDiscriminatorLoss
------------------------------------

.. autoclass:: AuxiliaryClassifierDiscriminatorLoss
    :members:

Feature Matching Loss
=====================

FeatureMatchingGeneratorLoss
----------------------------

.. autoclass:: FeatureMatchingGeneratorLoss
    :members:

Historical Averaging
====================

HistoricalAverageGeneratorLoss
------------------------------

.. autoclass:: HistoricalAverageGeneratorLoss
    :members:

HistoricalAverageDiscriminatorLoss
----------------------------------

.. autoclass:: HistoricalAverageDiscriminatorLoss
    :members:
