# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Example: One-pass calculation of feature mean, variance, and standard deviation
===============================================================================

This example shows how to calculate mean, variance, and standard deviation of log normalized
feature count data in one pass [1].

Example run::
    python examples/onepass_mean_var_std.py fit \
            --model.module scvid.module.OnePassMeanVarStdWithDefaults \
            --data.filenames \
            "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/benchmark_v1.{000..003}.h5ad" \
            --data.shard_size 10_000 --data.max_cache_size 2 --data.batch_size 10_000 \
            --data.shuffle true --data.num_workers 4 \
            --trainer.accelerator gpu --trainer.devices 1 --trainer.default_root_dir runs/onepass

**References:**

    1. *Algorithms for calculating variance*,
       (https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance)
"""

from lightning.pytorch.cli import LightningCLI

from scvid.data import DistributedAnnDataCollectionDataModule
from scvid.train.training_plan import DummyTrainingPlan


def main():
    LightningCLI(
        DummyTrainingPlan,
        DistributedAnnDataCollectionDataModule,
        trainer_defaults={"max_epochs": 1},  # one pass
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
