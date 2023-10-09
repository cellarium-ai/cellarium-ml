# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

r"""
Example: Geneformer
===================

This example shows how to fit feature count data to the Geneformer model [1].

Example run::

    python geneformer.py fit \
        --model.module cellarium.ml.module.GeneformerFromCLI \
        --data.filenames "gs://dsp-cellarium-cas-public/test-data/test_{0..3}.h5ad" \
        --data.shard_size 100 \
        --data.max_cache_size 2 \
        --data.batch_size 5 \
        --data.num_workers 1 \
        --trainer.accelerator gpu \
        --trainer.devices 1 \
        --trainer.default_root_dir runs/geneformer \
        --trainer.max_steps 10

**References:**

1. `Transfer learning enables predictions in network biology (Theodoris et al.)
   <https://www.nature.com/articles/s41586-023-06139-9>`_.
"""

from typing import Any

from jsonargparse import Namespace
from lightning.pytorch.cli import LightningCLI

from cellarium.ml.data import DistributedAnnDataCollectionDataModule
from cellarium.ml.train.training_plan import TrainingPlan


class _LightningCLIWithLinks(LightningCLI):
    """LightningCLI with custom argument linking."""

    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.var_names", "model.module.init_args.feature_schema", apply_on="instantiate")


def main(args: list[str] | dict[str, Any] | Namespace | None = None):
    """
    Args:
        args: Arguments to parse. If ``None`` the arguments are taken from ``sys.argv``.
    """
    _LightningCLIWithLinks(
        TrainingPlan,
        DistributedAnnDataCollectionDataModule,
        args=args,
    )


if __name__ == "__main__":
    main()
