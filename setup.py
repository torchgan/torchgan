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
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M
    )
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


PATH_ROOT = os.path.dirname(__file__)
VERSION = find_version("torchgan", "__init__.py")


def load_requirements(path_dir=PATH_ROOT, comment_char="#"):
    with open(os.path.join(path_dir, "requirements.txt"), "r") as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            ln = ln[: ln.index(comment_char)]
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs


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
    install_requires=load_requirements(PATH_ROOT),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords=["deep learning", "pytorch", "GAN", "AI"],
    python_requires=">=3.6",
    classifiers=[
        "Environment :: Console",
        "Natural Language :: English",
        # How mature is this project? Common values are
        #   3 - Alpha, 4 - Beta, 5 - Production/Stable
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Deep Learning",
        # Pick your license as you wish
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
