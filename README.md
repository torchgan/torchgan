<div align="center">

# TorchGAN

**Framework for easy and efficient training of GANs based on Pytorch**

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Downloads](https://pepy.tech/badge/torchgan)](https://pepy.tech/project/torchgan)
[![Downloads](https://pepy.tech/badge/torchgan/month)](https://pepy.tech/project/torchgan/month)
[![Downloads](https://pepy.tech/badge/torchgan/week)](https://pepy.tech/project/torchgan/week)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.02606/status.svg)](https://doi.org/10.21105/joss.02606)

[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://torchgan.readthedocs.io/en/stable/)
[![Latest Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://torchgan.readthedocs.io/en/latest/)
[![Codecov](https://codecov.io/gh/torchgan/torchgan/branch/master/graph/badge.svg)](https://codecov.io/gh/torchgan/torchgan)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/torchgan/torchgan/master)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/torchgan)
[![PyPI version](https://badge.fury.io/py/torchgan.svg)](https://badge.fury.io/py/torchgan)
</div>

TorchGAN is a [Pytorch](https://pytorch.org) based framework for designing and developing Generative Adversarial Networks. This framework has been designed to provide building blocks for popular GANs and also to allow customization for cutting edge research. Using TorchGAN's modular structure allows

* Trying out popular GAN models on your dataset.
* Plug in your new Loss Function, new Architecture, etc. with the traditional ones.
* Seamlessly visualize the training with a variety of logging backends.

| System / PyTorch Version | 1.8 | 1.9 | nightly |
| :---: | :---: | :---: | :---: |
| Linux py3.8 | ![CI Testing](https://github.com/torchgan/torchgan/workflows/CI%20Testing/badge.svg) | ![CI Testing](https://github.com/torchgan/torchgan/workflows/CI%20Testing/badge.svg) | ![CI Testing](https://github.com/torchgan/torchgan/workflows/CI%20Testing/badge.svg) | ![CI Testing](https://github.com/torchgan/torchgan/workflows/CI%20Testing/badge.svg) |
| Linux py3.9 | ![CI Testing](https://github.com/torchgan/torchgan/workflows/CI%20Testing/badge.svg) | ![CI Testing](https://github.com/torchgan/torchgan/workflows/CI%20Testing/badge.svg) | ![CI Testing](https://github.com/torchgan/torchgan/workflows/CI%20Testing/badge.svg) | ![CI Testing](https://github.com/torchgan/torchgan/workflows/CI%20Testing/badge.svg) |
| OSX py3.8 | ![CI Testing](https://github.com/torchgan/torchgan/workflows/CI%20Testing/badge.svg) | ![CI Testing](https://github.com/torchgan/torchgan/workflows/CI%20Testing/badge.svg) | ![CI Testing](https://github.com/torchgan/torchgan/workflows/CI%20Testing/badge.svg) | ![CI Testing](https://github.com/torchgan/torchgan/workflows/CI%20Testing/badge.svg) |
| OSX py3.9 | ![CI Testing](https://github.com/torchgan/torchgan/workflows/CI%20Testing/badge.svg) | ![CI Testing](https://github.com/torchgan/torchgan/workflows/CI%20Testing/badge.svg) | ![CI Testing](https://github.com/torchgan/torchgan/workflows/CI%20Testing/badge.svg) | ![CI Testing](https://github.com/torchgan/torchgan/workflows/CI%20Testing/badge.svg) |
| Windows py3.9 | ![CI Testing](https://github.com/torchgan/torchgan/workflows/CI%20Testing/badge.svg) | ![CI Testing](https://github.com/torchgan/torchgan/workflows/CI%20Testing/badge.svg) | ![CI Testing](https://github.com/torchgan/torchgan/workflows/CI%20Testing/badge.svg) | ![CI Testing](https://github.com/torchgan/torchgan/workflows/CI%20Testing/badge.svg) |
| Windows py3.9 | ![CI Testing](https://github.com/torchgan/torchgan/workflows/CI%20Testing/badge.svg) | ![CI Testing](https://github.com/torchgan/torchgan/workflows/CI%20Testing/badge.svg) | ![CI Testing](https://github.com/torchgan/torchgan/workflows/CI%20Testing/badge.svg) | ![CI Testing](https://github.com/torchgan/torchgan/workflows/CI%20Testing/badge.svg) |

### Installation

Using pip (for stable release):

```bash
  $ pip install torchgan
```

Using pip (for latest master):

```bash
  $ pip install git+https://github.com/torchgan/torchgan.git
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

### Tutorials

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/torchgan/torchgan/master)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/torchgan)

The `tutorials` directory contain a set of tutorials to get you started with torchgan. These tutorials can be run using Google Colab or Binder. It is highly recommended that you follow the tutorials in the following order.

1. Introductory Tutorials:
    - [Tutorial 1: Introduction to TorchGAN](https://github.com/torchgan/torchgan/blob/master/tutorials/Tutorial%201.%20Introduction%20to%20TorchGAN.ipynb)
    - [Tutorial 2: Custom Loss Functions](https://github.com/torchgan/torchgan/blob/master/tutorials/Tutorial%202.%20Custom%20Loss%20Functions.ipynb)
2. Intermediate Tutorials:
    - [Tutorial 3: CycleGAN](https://github.com/torchgan/torchgan/blob/master/tutorials/Tutorial%203.%20CycleGAN.ipynb)
    - [Tutorial 4: Self Attention GAN](https://github.com/torchgan/torchgan/blob/master/tutorials/Tutorial%204.%20Self%20Attention%20GAN.ipynb)
3. Advanced Tutorials:
    - [Tutorial 5: Adversarial Autoencoder](https://github.com/torchgan/torchgan/blob/master/tutorials/Tutorial%205.%20Adversarial%20Autoencoder.ipynb)

### Supporting and Citing

This software was developed as part of academic research. If you would like to help support it, please star the repository. If you use this software as part of your research, teaching, or other activities, we would be grateful if you could cite the following:

```
@article{Pal2021,
  doi = {10.21105/joss.02606},
  url = {https://doi.org/10.21105/joss.02606},
  year = {2021},
  publisher = {The Open Journal},
  volume = {6},
  number = {66},
  pages = {2606},
  author = {Avik Pal and Aniket Das},
  title = {TorchGAN: A Flexible Framework for GAN Training and Evaluation},
  journal = {Journal of Open Source Software}
}
```

List of publications & submissions using TorchGAN (please open a pull request to add missing entries):

* [Can GAN-Generated Network Traffic be used to Train Traffic Anomaly Classifiers?](https://ieeexplore.ieee.org/abstract/document/9284901?casa_token=9bEvJ3COOXMAAAAA:pz8TsMSrecv_ip1t9rEI7tYn_S_AQyZE_UrgYZ61vX_3Clu0Y17pFTUEpclAcBja13pPEqOsxypp)
* [Ward2ICU: A Vital Signs Dataset of Inpatients from the General Ward](https://arxiv.org/abs/1910.00752)

### Contributing

We appreciate all contributions. If you are planning to contribute bug-fixes, please do so without any further discussion. If you plan to contribute new features, utility functions or extensions, please first open an issue and discuss the feature with us. For more detailed guidelines head over to the official documentation.

### Contributors

This package has been developed by
* Avik Pal (@avik-pal)
* Aniket Das (@Aniket1998)

This project exists thanks to all the people who contribute.

<a href="https://github.com/torchgan/torchgan/graphs/contributors"><img src="https://opencollective.com/torchgan/contributors.svg?width=890&button=false" /></a>
