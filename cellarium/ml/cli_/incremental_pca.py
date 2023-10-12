# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

r"""
Example: Incremental PCA
========================

This example shows how to fit feature count data to incremental PCA
model [1, 2].

Example run::

    cellarium-incremental-pca fit \
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

from lightning.pytorch.cli import ArgsType

from cellarium.ml.cli.lightning_cli import lightning_cli_factory


def main(args: ArgsType = None) -> None:
    """
    Args:
        args: Arguments to parse. If ``None`` the arguments are taken from ``sys.argv``.
    """
    cli = lightning_cli_factory(
        "cellarium.ml.module.IncrementalPCAFromCLI",
        link_arguments=[("data.n_vars", "model.module.init_args.g_genes")],
        trainer_defaults={
            "max_epochs": 1,  # one pass
            "strategy": {
                "class_path": "lightning.pytorch.strategies.DDPStrategy",
                "init_args": {"broadcast_buffers": False},
            },
        },
    )
    cli(args=args)


if __name__ == "__main__":
    main()
