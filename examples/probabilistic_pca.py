# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Example: Probabilistic PCA
==========================

This example shows how to fit feature count data to probabilistic PCA
model [1].

There are two flavors of probabilistic PCA model that are available:

1. "marginalized" - latent variable ``z`` is marginalized out [1].
2. "linear_vae" - latent variable ``z`` has a diagonal Gaussian distribution [2].

Example run::
    python examples/probabilistic_pca.py fit \
            --model.module.class_path scvid.module.ProbabilisticPCAWithDefaults \
            --model.module.init_args.mean_var_std_ckpt_path \
            "runs/onepass/lightning_logs/version_0/checkpoints/epoch=0-step=4.ckpt" \
            --data.filenames \
            "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/benchmark_v1.{000..003}.h5ad" \
            --data.shard_size 10_000 --data.max_cache_size 2 --data.batch_size 10_000 \
            --data.shuffle true --data.num_workers 4 \
            --trainer.accelerator gpu --trainer.devices 1 --trainer.max_steps 1000 \
            --trainer.default_root_dir runs/ppca

**References:**

    1. *Probabilistic Principal Component Analysis*,
       Tipping, Michael E., and Christopher M. Bishop. 1999.
       (https://www.robots.ox.ac.uk/~cvrg/hilary2006/ppca.pdf)
    2. *Understanding Posterior Collapse in Generative Latent Variable Models*,
       James Lucas, George Tucker, Roger Grosse, Mohammad Norouzi. 2019.
       (https://openreview.net/pdf?id=r1xaVLUYuE)
"""

from lightning.pytorch.cli import LightningCLI

from scvid.data import DistributedAnnDataCollectionDataModule
from scvid.train import PyroTrainingPlan


class PPCALightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "data.n_obs", "model.module.init_args.n_cells", apply_on="instantiate"
        )
        parser.link_arguments(
            "data.n_vars", "model.module.init_args.g_genes", apply_on="instantiate"
        )


def main():
    PPCALightningCLI(
        PyroTrainingPlan,
        DistributedAnnDataCollectionDataModule,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    #  parser.add_argument(
    #      "-lr", "--learning_rate", default=0.1, type=float, help="learning rate"
    #  )
    #  parser.add_argument("--log_every_n_steps", type=int, default=1)
    main()
