# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

r"""
Example: Incremental PCA
========================

This example shows how to fit feature count data to incremental PCA
model [1, 2].

Example run::

    python examples/incremental_pca.py fit \
        --model.module cellarium.ml.module.IncrementalPCAFromCLI \
        --model.module.init_args.k_components 50 \
        --data.filenames "gs://dsp-cellarium-cas-public/test-data/test_{0..3}.h5ad" \
        --data.shard_size 100 \
        --data.max_cache_size 2 \
        --data.batch_size 100 \
        --data.num_workers 4 \
        --trainer.accelerator gpu \
        --trainer.devices 1 \
        --trainer.default_root_dir runs/ipca \
        --trainer.callbacks cellarium.ml.callbacks.ModuleCheckpoint

**References:**

1. `A Distributed and Incremental SVD Algorithm for Agglomerative Data Analysis on Large Networks (Iwen et al.)
   <https://users.math.msu.edu/users/iwenmark/Papers/distrib_inc_svd.pdf>`_.
2. `Incremental Learning for Robust Visual Tracking (Ross et al.)
   <https://www.cs.toronto.edu/~dross/ivt/RossLimLinYang_ijcv.pdf>`_.
"""

from lightning.pytorch.cli import LightningCLI

from cellarium.ml.data import DistributedAnnDataCollectionDataModule
from cellarium.ml.train.training_plan import TrainingPlan


class _LightningCLIWithLinks(LightningCLI):
    """LightningCLI with custom argument linking."""

    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.n_vars", "model.module.init_args.g_genes", apply_on="instantiate")


def main(args=None):
    _LightningCLIWithLinks(
        TrainingPlan,
        DistributedAnnDataCollectionDataModule,
        trainer_defaults={"max_epochs": 1},  # one pass
        args=args,
    )


if __name__ == "__main__":
    main()
