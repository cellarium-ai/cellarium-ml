# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

r"""
Example: Incremental PCA
========================

This example shows how to fit feature count data to incremental PCA
model [1, 2].

Example run::

    python examples/incremental_pca.py fit \
        --model.module scvid.module.IncrementalPCAFromCLI \
        --model.module.init_args.k_components 50 \
        --data.filenames "gs://dsp-cellarium-cas-public/test-data/benchmark_v1.{000..003}.h5ad" \
        --data.shard_size 10_000 --data.max_cache_size 2 --data.batch_size 10_000 \
        --data.num_workers 4 \
        --trainer.accelerator gpu --trainer.devices 1 --trainer.default_root_dir runs/ipca \
        --trainer.callbacks scvid.callbacks.ModuleCheckpoint

**References:**

    1. *A Distributed and Incremental SVD Algorithm for Agglomerative Data Analysis on Large Networks*,
       M. A. Iwen, B. W. Ong
       (https://users.math.msu.edu/users/iwenmark/Papers/distrib_inc_svd.pdf)
    2. *Incremental Learning for Robust Visual Tracking*,
       D. Ross, J. Lim, R.-S. Lin, M.-H. Yang
       (https://www.cs.toronto.edu/~dross/ivt/RossLimLinYang_ijcv.pdf)
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
        trainer_defaults={"max_epochs": 1},  # one pass
    )


if __name__ == "__main__":
    main()
