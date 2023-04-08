# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Example: One-pass calculation of feature mean, variance, and standard deviation
===============================================================================

This example shows how to calculate mean, variance, and standard deviation of log normalized
feature count data in one pass [1].

Example run::
    python examples/onepass_mean_var_std.py \
            --filenames gs://dsp-cell-annotation-service/benchmark_v1/benchmark_v1.{000..324}.h5ad \
            --accelerator gpu --devices 1 --num_workers 4 \
            --default_root_dir runs/onepass

**References:**

    1. *Algorithms for calculating variance*,
       (https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance)
"""

import argparse

import lightning.pytorch as pl
import torch

from scvid.data import (
    DistributedAnnDataCollection,
    IterableDistributedAnnDataCollectionDataset,
)
from scvid.data.util import collate_fn
from scvid.module.onepass_mean_var_std import OnePassMeanVarStd
from scvid.train.training_plan import DummyTrainingPlan
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
    onepass = OnePassMeanVarStd(transform=transform)
    plan = DummyTrainingPlan(onepass)

    # train
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=1,  # one pass
        strategy=args.strategy,
        default_root_dir=args.default_root_dir,
    )
    trainer.fit(plan, train_dataloaders=data_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OnePassMeanVarStd example")
    parser.add_argument("--filenames", type=str, help="path to anndata files")
    parser.add_argument("--batch_size", default=10_000, type=int, help="batch size")
    parser.add_argument("--num-workers", default=4, type=int, help="number of workers")
    # Trainer args
    parser.add_argument("--accelerator", type=str, default="cpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--default_root_dir", type=str, default="runs/onepass")
    args = parser.parse_args()

    main(args)
