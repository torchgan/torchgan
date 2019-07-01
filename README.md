# TorchGAN

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Build Status](https://travis-ci.org/torchgan/torchgan.svg?branch=master)](https://travis-ci.org/torchgan/torchgan)
[![Build status](https://ci.appveyor.com/api/projects/status/lqo3bbsew83839c0/branch/master?svg=true)](https://ci.appveyor.com/project/avik-pal/torchgan/branch/master)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE)
[![Slack](https://img.shields.io/badge/chat-on%20slack-yellow.svg)](https://join.slack.com/t/torchgan/shared_invite/enQtNDkyMTQ2ODAyMzczLWEyZjc1ZDdmNTc3ZmNiODFmMmY2YjM2OTZmZTRlOTc3YWE5MTliZTBkZTkwNzQ2MDIwZmI0MGRjYjQwYTczMzQ)
[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://torchgan.readthedocs.io/en/stable/)
[![Latest Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://torchgan.readthedocs.io/en/latest/)
[![codecov](https://codecov.io/gh/torchgan/torchgan/branch/master/graph/badge.svg)](https://codecov.io/gh/torchgan/torchgan)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/torchgan/torchgan/master)
[![PyPI version](https://badge.fury.io/py/torchgan.svg)](https://badge.fury.io/py/torchgan)

TorchGAN is a [Pytorch](https://pytorch.org) based framework for designing and developing Generative Adversarial Networks. This framework has been designed to provide building blocks for popular GANs and also to allow customization for cutting edge research. Using TorchGAN's modular structure allows

* Trying out popular GAN models on your dataset.
* Plug in your new Loss Function, new Architecture, etc. with the traditional ones.
* Seamlessly visualize the training with a variety of logging backends.

### Installation

Using pip (for stable release):

```bash
  $ pip3 install torchgan
```

Using pip (for latest master):

```bash
  $ pip3 install git+https://github.com/torchgan/torchgan.git
```

From source:

```bash
  $ git clone https://github.com/torchgan/torchgan.git
  $ cd torchgan
  $ python setup.py install
```

### Documentation

The documentation is available [here](https://torchgan.readthedocs.io/en/latest/)

The documentation for this package can be generated locally.

```bash
  $ git clone https://github.com/torchgan/torchgan.git
  $ cd torchgan/docs
  $ pip install -r requirements.txt
  $ make html
```

Now open the corresponding file from `build` directory.

### Contributing

We appreciate all contributions. If you are planning to contribute bug-fixes, please do so without any further discussion. If you plan to contribute new features, utility functions or extensions, please first open an issue and discuss the feature with us. For more detailed guidelines head over to the official documentation.

### Tutorials

The `examples` directory contain a set of tutorials to get you started with torchgan. Some of these notebooks are available on Google Colab (*they are linked in the tutorials themselves*). Additionally, these tutorials can be tried out using the binder link provided.

### Contributors

This package has been developed by
* Avik Pal (@avik-pal)
* Aniket Das (@Aniket1998)

We appreciate all contributors. For a full list of contributors go [here](https://github.com/torchgan/torchgan/graphs/contributors).
