# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

r"""
Example: One-pass calculation of feature median using t-digest statistics
=========================================================================

This example shows how to calculate non-zero median of normalized feature count
data in one pass [1].

Example run::

    python examples/tdigest.py fit \
        --model.module scvid.module.TDigestFromCLI \
        --data.filenames "gs://dsp-cellarium-cas-public/test-data/benchmark_v1.{000..003}.h5ad" \
        --data.shard_size 10_000 --data.max_cache_size 2 --data.batch_size 10_000 \
        --data.num_workers 4 \
        --trainer.accelerator cpu --trainer.devices 1 --trainer.default_root_dir runs/tdigest \
        --trainer.callbacks scvid.callbacks.ModuleCheckpoint

**References:**

    1. *Computing Extremely Accurate Quantiles Using T-Digests*,
       Dunning, Ted, and Otmar Ertl.
       (https://github.com/tdunning/t-digest/blob/master/docs/t-digest-paper/histo.pdf)
"""

from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from scvid.data import DistributedAnnDataCollectionDataModule
from scvid.train.training_plan import TrainingPlan


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
