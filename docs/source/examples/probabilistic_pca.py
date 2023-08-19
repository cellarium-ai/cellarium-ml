# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

r"""
Example: Probabilistic PCA
==========================

This example shows how to fit feature count data to probabilistic PCA
model [1].

There are two flavors of probabilistic PCA model that are available:

1. ``marginalized`` - latent variable ``z`` is marginalized out [1]. Marginalized
   model provides a closed-form solution for the marginal log-likelihood.
   Closed-form solution for the marginal log-likelihood has reduced
   variance compared to the ``linear_vae`` model.
2. ``linear_vae`` - latent variable ``z`` has a diagonal Gaussian distribution [2].
   Training a linear VAE with variational inference recovers a uniquely identifiable
   global maximum  corresponding to the principal component directions.
   The global maximum of the ELBO objective for the linear VAE  is identical
   to the global maximum for the marginal log-likelihood of probabilistic PCA.

Example run::

    python examples/probabilistic_pca.py fit \
        --model.module.class_path scvid.module.ProbabilisticPCAFromCLI \
        --model.module.init_args.mean_var_std_ckpt_path \
        "runs/onepass/lightning_logs/version_0/checkpoints/module_checkpoint.pt" \
        --data.filenames "gs://dsp-cellarium-cas-public/test-data/benchmark_v1.{000..003}.h5ad" \
        --data.shard_size 10_000 --data.max_cache_size 2 --data.batch_size 10_000 \
        --data.shuffle true --data.num_workers 4 \
        --trainer.accelerator gpu --trainer.devices 1 --trainer.max_steps 1000 \
        --trainer.default_root_dir runs/ppca \
        --trainer.callbacks scvid.callbacks.VarianceMonitor \
        --trainer.callbacks.mean_var_std_ckpt_path \
        "runs/onepass/lightning_logs/version_0/checkpoints/module_checkpoint.pt"

**References:**

    1. *Probabilistic Principal Component Analysis*,
       Tipping, Michael E., and Christopher M. Bishop. 1999.
       (https://www.robots.ox.ac.uk/~cvrg/hilary2006/ppca.pdf)
    2. *Understanding Posterior Collapse in Generative Latent Variable Models*,
       James Lucas, George Tucker, Roger Grosse, Mohammad Norouzi. 2019.
       (https://openreview.net/pdf?id=r1xaVLUYuE)
"""

from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from scvid.data import DistributedAnnDataCollectionDataModule
from scvid.train import TrainingPlan


class _LightningCLIWithLinks(LightningCLI):
    """LightningCLI with custom argument linking."""

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments(
            "data.n_obs", "model.module.init_args.n_cells", apply_on="instantiate"
        )
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
