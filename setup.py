# coding: utf-8
## To check what packages are being included in the wheel:
## python -c "from setuptools import setup, find_packages; print(find_packages())"
"""Setup Pytorch-VAE package."""

from setuptools import setup, find_packages

setup(
    name='pytorch_vae',
    version='0.0.1',
    description='Fork of AntixK/PyTorch-VAE',
    author='',
    author_email='',
    url='https://github.com/AntixK',
    install_requires=[],
    packages=find_packages(),
    package_data={},
)