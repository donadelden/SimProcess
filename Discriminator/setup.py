#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="simprocess",
    version="1.0.0",
    description="Signal Analysis and Classification Framework",
    author="PLACEHOLDER",
    author_email="PLACEHOLDER@math.unipd.it",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "tsfresh>=0.18.0",
        "scipy>=1.5.0",
        "joblib>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "simprocess=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
)