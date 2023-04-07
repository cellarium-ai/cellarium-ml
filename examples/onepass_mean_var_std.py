# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Example: One-pass calculation of feature mean, variance, and standard deviation
===============================================================================

This example shows how to calculate mean, variance, and standard deviation of log normalized
feature count data in one pass [1].

Example run::
    python examples/onepass_mean_var_std.py \
            --filenames gs://dsp-cell-annotation-service/benchmark_v1/benchmark_v1.{000..324}.h5ad \
            --accelerator gpu --devices 1 --num_workers 4 \
            --default_root_dir runs/onepass

**References:**

    1. *Algorithms for calculating variance*,
       (https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance)
"""

from lightning.pytorch.cli import LightningCLI

from scvid.data import DistributedAnnDataCollectionDataModule
from scvid.train.training_plan import DummyTrainingPlan


def main():
    cli = LightningCLI(DummyTrainingPlan, DistributedAnnDataCollectionDataModule)


if __name__ == "__main__":
    main()
