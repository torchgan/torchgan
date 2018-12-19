================
torchgan.logging
================

.. currentmodule:: torchgan.logging

This subpackage provides strong visualization capabilities using a variety of Backends.
It is strongly integrated with the Trainer. The ``Logger`` supports a variety of
configurations and customizations.

.. contents::
    :local:

.. note::
    The Logger API is currently deeply integrated with the ``Trainer`` and hence might not
    be a very pleasant thing to use externally. However, work is being done to make them
    as much independent as possible and support extendibility of the Logger. Hence, this is
    expected to improve in the future.

Backends
========

Currently available backends are:

1. **TensorboardX**:
    To enable this set the ``TENSORBOARD_LOGGING`` to 1. If the package is pre-installed
    on your system, this variable is enabled by default.

    If you want to disable this then :code:`os.environ["TENSORBOARD_LOGGING"] = "0"`. Make sure to do it
    before loading torchgan.

    Once the logging begins, you need to start a tensorboard server using this code :code:`tensorboard
    --logdir runs`.

2. **Visdom**:
    To enable this set the ``VISDOM_LOGGING`` to 1. If the package is pre-installed
    on your system, this variable is enabled by default.

    If you want to disable this then :code:`os.environ["VISDOM_LOGGING"] = "0"`. We recommend using
    visdom if you need to save your plots. In general tensorboard support is better in terms of the
    image display.

    .. warning::
        If this package is present and **VISDOM_LOGGING** is set to 1, then a server must be started
        using the command `python -m visdom.server` before the Training is started. Otherwise the
        code will simply crash.

3. **Console**:
    The details of training are printed on the console. This is enabled by default but can be turned
    off by :code:`os.environ["CONSOLE_LOGGING"] = "0"`.

Add more backends for visualization is a work-in-progress.

.. note::
    It is the responsibility of the user to install the necessary packages needed for visualization.
    If the necessary packages are missing the logging will not occur or if the user trys to force it
    the program will terminate with an error message.

.. note::
    It is recommended to use only **1 logging service** (apart from the **Console**). Using multiple
    Logging services might affect the training time. It is recommended to use **Visdom** only if the
    plots are to be downloaded easily.

Logger
======

.. autoclass:: Logger
    :members:

Visualization
=============

Visualize
---------

.. autoclass:: Visualize
    :members:

LossVisualize
-------------

.. autoclass:: LossVisualize
    :members:

GradientVisualize
-----------------

.. autoclass:: GradientVisualize
    :members:

MetricVisualize
---------------

.. autoclass:: MetricVisualize
    :members:

ImageVisualize
--------------

.. autoclass:: ImageVisualize
    :members:
