# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

r"""
Example: One-pass calculation of feature median using t-digest statistics
=========================================================================

This example shows how to calculate non-zero median of normalized feature count
data in one pass [1].

Example run::

    tdigest fit \
        --data.filenames "gs://dsp-cellarium-cas-public/test-data/test_{0..3}.h5ad" \
        --data.shard_size 100 \
        --data.max_cache_size 2 \
        --data.batch_size 100 \
        --data.num_workers 4 \
        --trainer.accelerator cpu \
        --trainer.devices 1 \
        --trainer.default_root_dir runs/tdigest \
        --trainer.callbacks cellarium.ml.callbacks.ModuleCheckpoint

**References:**

1. `Computing Extremely Accurate Quantiles Using T-Digests (Dunning et al.)
   <https://github.com/tdunning/t-digest/blob/master/docs/t-digest-paper/histo.pdf>`_.
"""

from lightning.pytorch.cli import ArgsType

from cellarium.ml.cli.lightning_cli import lightning_cli_factory


def main(args: ArgsType = None) -> None:
    """
    Args:
        args: Arguments to parse. If ``None`` the arguments are taken from ``sys.argv``.
    """
    cli = lightning_cli_factory(
        "cellarium.ml.module.TDigestFromCLI",
        link_arguments=[("data.n_vars", "model.module.init_args.g_genes")],
        trainer_defaults={
            "max_epochs": 1,  # one pass
        },
    )
    cli(args=args)


if __name__ == "__main__":
    main()
