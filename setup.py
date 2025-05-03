#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="semantic-calculator",
    version="0.1.0",
    description="Semantic operations tool for vector and emoji calculations",
    author="Claude & Rob",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "sentence-transformers>=2.2.0",
        "torch>=1.10.0",
        "matplotlib>=3.4.0",
        "umap-learn>=0.5.2",
        "plotly>=5.5.0",
        "seaborn>=0.11.2",
    ],
    entry_points={
        "console_scripts": [
            "semantic-calculator=semantic_calculator.__main__:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)
