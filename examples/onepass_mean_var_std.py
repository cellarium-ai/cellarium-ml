# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

r"""
Example: One-pass calculation of feature mean, variance, and standard deviation
===============================================================================

This example shows how to calculate mean, variance, and standard deviation of log normalized
feature count data in one pass [1].

Example run::

    python examples/onepass_mean_var_std.py fit \
        --model.module cellarium.ml.module.OnePassMeanVarStdFromCLI \
        --data.filenames "gs://dsp-cellarium-cas-public/test-data/benchmark_v1.{000..003}.h5ad" \
        --data.shard_size 100 --data.max_cache_size 2 --data.batch_size 100 \
        --data.num_workers 4 \
        --trainer.accelerator gpu --trainer.devices 1 --trainer.default_root_dir runs/onepass \
        --trainer.callbacks cellarium.ml.callbacks.ModuleCheckpoint

**References:**

    1. *Algorithms for calculating variance*,
       (https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance)
"""

from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from cellarium.ml.data import DistributedAnnDataCollectionDataModule
from cellarium.ml.train.training_plan import TrainingPlan


class _LightningCLIWithLinks(LightningCLI):
    """LightningCLI with custom argument linking."""

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments("data.n_vars", "model.module.init_args.g_genes", apply_on="instantiate")


def main():
    _LightningCLIWithLinks(
        TrainingPlan,
        DistributedAnnDataCollectionDataModule,
        trainer_defaults={"max_epochs": 1},  # one pass
    )


if __name__ == "__main__":
    main()
