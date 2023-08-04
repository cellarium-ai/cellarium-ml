# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Example: Geneformer
===================

This example shows how to fit feature count data to Geneformer model [1].

Example run::
    python examples/geneformer.py fit \
        --model.module scvid.module.geneformer.Geneformer \
        --data.filenames "gs://dsp-cellarium-cas-public/test-data/benchmark_v1.{000..003}.h5ad" \
        --data.shard_size 10_000 --data.max_cache_size 2 --data.batch_size 10_000 \
        --data.num_workers 4 \
        --trainer.accelerator gpu --trainer.devices 1 --trainer.default_root_dir runs/ipca \
        --trainer.callbacks scvid.callbacks.ModuleCheckpoint

**References:**

    1. *Transfer learning enables predictions in network biology*,
       C. V. Theodoris, L. Xiao, A. Chopra, M. D. Chaffin, Z. R. Al Sayed,
       M. C. Hill, H. Mantineo, E. Brydon, Z. Zeng, X. S. Liu & P. T. Ellinor
       (https://www.nature.com/articles/s41586-023-06139-9)
"""

from lightning.pytorch.cli import LightningCLI

from scvid.data import DistributedAnnDataCollectionDataModule
from scvid.train.training_plan import TrainingPlan


class _LightningCLIWithLinks(LightningCLI):
    """LightningCLI with custom argument linking."""

    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "data.n_vars", "model.module.init_args.g_genes", apply_on="instantiate"
        )


def main():
    _LightningCLIWithLinks(
        TrainingPlan,
        DistributedAnnDataCollectionDataModule,
    )


if __name__ == "__main__":
    main()
