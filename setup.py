# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os

import setuptools

import versioneer

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
VERSION = """
# This file is auto-generated with the version information during setup.py installation.
__version__ = '{}'
"""

with open("README.rst", "r") as fh:
    long_description = fh.read()

# tests
LINT_REQUIRE = [
    "black[jupyter]",
    "flake8",
    "isort",
]
MYPY_REQUIRE = [
    "mypy",
]
TEST_REQUIRE = [
    "pytest",
    "pytest-xdist",
    "scikit-learn",
    "tensorboard",
]

setuptools.setup(
    name="scvi-distributed",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Distributed single-cell data analysis",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://cellarium.ai",
    project_urls={
        "Source": "https://github.com/cellarium-ai/scvi-distributed",
    },
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        "anndata",
        "google-cloud-storage",
        "boltons",
        "braceexpand",
        "jsonargparse[signatures]",
        "pyro-ppl",
        "lightning>=2.0.0",
        "torch>=2.0.0",
    ],
    extras_require={
        "lint": LINT_REQUIRE,
        "mypy": MYPY_REQUIRE,
        "test": TEST_REQUIRE,
        "dev": LINT_REQUIRE + MYPY_REQUIRE + TEST_REQUIRE,
    },
    keywords="scvi-tools anndata distributed",
    license="Apache 2.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.10",
)
