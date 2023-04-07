# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Example: Probabilistic PCA
==========================

This example shows how to fit feature count data to probabilistic PCA
model [1].

There are three flavors of probabilistic PCA model that are available:

1. "marginalized" - latent variable ``z`` is marginalized out.
2. "multivariate_normal" - latent variable ``z`` has a multivariate Gaussian distribution.
3. "diagonal_normal" - latent variable ``z`` has a diagonal Gaussian distribution.

Example run::
    python examples/probabilistic_pca.py \
            --filenames gs://dsp-cell-annotation-service/benchmark_v1/benchmark_v1.{000..324}.h5ad \
            --accelerator gpu --devices 1 --max_steps 1000 --num_workers 4 \
            --ppca_flavor marginalized --log_every_n_steps 1 --default_root_dir runs/ppca

**References:**

    1. *Probabilistic Principal Component Analysis*,
       Tipping, Michael E., and Christopher M. Bishop. 1999.
       (https://www.robots.ox.ac.uk/~cvrg/hilary2006/ppca.pdf)
"""

import argparse

import lightning.pytorch as pl
import torch
from lightning.pytorch.cli import LightningCLI

from scvid.callbacks import VarianceMonitor
from scvid.data import (
    DistributedAnnDataCollection,
    IterableDistributedAnnDataCollectionDataset,
)
from scvid.data.util import collate_fn
from scvid.module import ProbabilisticPCAPyroModule
from scvid.train import PyroTrainingPlan
from scvid.transforms import ZScoreLog1pNormalize


class _ProbabilisticPCA(ProbabilisticPCAPyroModule):
    def __init__(
        self,
        n_cells: int,
        g_genes: int,
        k_components: int,
        ppca_flavor: str,
        mean_var_std_ckpt_path: Optional[str] = None,
        W_init_scale_relative: float = 0.5,
        sigma_init_scale_relative: float = 0.5,
        seed: int = 0,
    ):
        super().__init__(
            n_cells=n_cells,
            g_genes=g_genes,
            k_components=k_components,
            ppca_flavor=ppca_flavor,
            mean_g=mean_g,
            W_init_scale=W_init_scale,
            sigma_init_scale=sigma_init_scale,
            seed=seed,
            transform=transform,
        )


def main():
    # setup model and training plan
    #  transform = ZScoreLog1pNormalize(
    #      mean_g=0, std_g=None, perform_scaling=False, target_count=10_000
    #  )
    #  onepass = torch.load("onepass.pt")
    #  w = torch.sqrt(0.5 * onepass["var"].sum() / (dadc.n_vars * args.num_components)).item()
    #  s = torch.sqrt(0.5 * onepass["var"].sum() / dadc.n_vars).item()
    #  ppca = ProbabilisticPCAPyroModule(
    #      n_cells=dadc.n_obs,
    #      g_genes=dadc.n_vars,
    #      k_components=args.num_components,
    #      ppca_flavor=args.ppca_flavor,
    #      mean_g=onepass["mean"].cuda(),  # learned
    #      transform=transform,
    #      W_init_scale=w,
    #      sigma_init_scale=s,
    #  )
    #  plan = PyroTrainingPlan(ppca, optim_kwargs={"lr": args.learning_rate})
    cli = LightningCLI(PyroTrainingPlan, DataModule)

    # train
    #  trainer = pl.Trainer(
    #      accelerator=args.accelerator,
    #      devices=args.devices,
    #      max_steps=args.max_steps,
    #      log_every_n_steps=args.log_every_n_steps,
    #      strategy=args.strategy,
    #      default_root_dir=args.default_root_dir,
    #  )
    #  trainer.fit(plan, train_dataloaders=data_loader, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    #  parser = argparse.ArgumentParser(description="Probabilistic PCA example")
    #  parser.add_argument("--filenames", type=str, help="path to anndata files")
    #  parser.add_argument("--batch_size", default=10_000, type=int, help="batch size")
    #  parser.add_argument("--num_workers", default=4, type=int, help="number of workers")
    #  parser.add_argument(
    #      "-lr", "--learning_rate", default=0.1, type=float, help="learning rate"
    #  )
    #  parser.add_argument(
    #      "--num_components", default=256, type=int, help="number of PCA components"
    #  )
    #  parser.add_argument(
    #      "--ckpt_path",
    #      type=str,
    #      help="path of the checkpoint from which training is resumed",
    #  )
    #  parser.add_argument(
    #      "--ppca_flavor",
    #      default="marginalized",
    #      type=str,
    #      choices=["marginalized", "diagonal_normal", "multivariate_normal"],
    #      help="probabilistic PCA flavor",
    #  )
    #  # Trainer args
    #  parser.add_argument("--accelerator", type=str, default="gpu")
    #  parser.add_argument("--devices", type=int, default=1)
    #  parser.add_argument("--max_steps", type=int, default=1000)
    #  parser.add_argument("--log_every_n_steps", type=int, default=1)
    #  parser.add_argument("--strategy", type=str, default="auto")
    #  parser.add_argument("--default_root_dir", type=str, default="runs/ppca")
    #  args = parser.parse_args()

    main()
