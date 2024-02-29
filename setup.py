#!/usr/bin/env python

from setuptools import find_packages, setup

with open("README") as f:
    readme = f.read()

setup(
    name="particle_fm",
    version="1.0.0",
    description="Easily train and evaluate multiple generative models on various particle physics datasets",
    author="Cedric Ewen",
    author_email="chessdric@gmail.com",
    url="https://github.com/ewencedr/Particle-FM",
    long_description=readme,
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
)
