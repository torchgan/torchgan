#!/usr/bin/env python
import io
import os
import re
import shutil
import sys

from pkg_resources import DistributionNotFound, get_distribution
from setuptools import find_packages, setup


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8"),
    ) as fp:
        return fp.read()


def get_dist(pkgname):
    try:
        return get_distribution(pkgname)
    except DistributionNotFound:
        return None


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


VERSION = find_version("torchgan", "__init__.py")

requirements = ["numpy", "pillow>=6.2.0", "fastprogress==0.1.20"]

setup(
    # Metadata
    name="torchgan",
    version=VERSION,
    author="Avik Pal & Aniket Das",
    author_email="avikpal@cse.iitk.ac.in",
    url="https://github.com/torchgan/torchgan",
    description="Research Framework for easy and efficient training of GANs based on Pytorch",
    license="MIT",
    # Package info
    packages=find_packages(exclude=("test",)),
    zip_safe=True,
    install_requires=requirements,
)
