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

import pytorch_lightning as pl
import torch

from scvid.data import (
    DistributedAnnDataCollection,
    IterableDistributedAnnDataCollectionDataset,
)
from scvid.data.util import collate_fn
from scvid.module import ProbabilisticPCAPyroModule
from scvid.train import PyroTrainingPlan
from scvid.transforms import ZScoreLog1pNormalize


def main(args):
    # data loader
    dadc = DistributedAnnDataCollection(
        filenames=args.filenames,
        shard_size=10_000,
        max_cache_size=2,
    )
    dataset = IterableDistributedAnnDataCollectionDataset(
        dadc, batch_size=args.batch_size, shuffle=True
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # setup model and training plan
    transform = ZScoreLog1pNormalize(
        mean_g=0, std_g=None, perform_scaling=False, target_count=10_000
    )
    ppca = ProbabilisticPCAPyroModule(
        n_cells=dadc.n_obs,
        g_genes=dadc.n_vars,
        k_components=args.num_components,
        ppca_flavor=args.ppca_flavor,
        mean_g=None,  # learned
        transform=transform,
    )
    plan = PyroTrainingPlan(ppca, optim_kwargs={"lr": args.learning_rate})

    # train
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_steps=args.max_steps,
        log_every_n_steps=args.log_every_n_steps,
        strategy=args.strategy,
        default_root_dir=args.default_root_dir,
    )
    trainer.fit(plan, train_dataloaders=data_loader, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probabilistic PCA example")
    parser.add_argument("--filenames", type=str, help="path to anndata files")
    parser.add_argument("--batch_size", default=10_000, type=int, help="batch size")
    parser.add_argument("--num_workers", default=4, type=int, help="number of workers")
    parser.add_argument(
        "-lr", "--learning_rate", default=0.1, type=float, help="learning rate"
    )
    parser.add_argument(
        "--num_components", default=256, type=int, help="number of PCA components"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="path of the checkpoint from which training is resumed",
    )
    parser.add_argument(
        "--ppca_flavor",
        default="marginalized",
        type=str,
        choices=["marginalized", "diagonal_normal", "multivariate_normal"],
        help="probabilistic PCA flavor",
    )
    # Trainer args
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--log_every_n_steps", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="ddp")
    parser.add_argument("--default_root_dir", type=str, default="runs/ppca")
    args = parser.parse_args()

    main(args)
