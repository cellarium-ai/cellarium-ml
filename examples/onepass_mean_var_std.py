"""
Example: One Pass Mean Var Std
==============================

This example shows how to run calculate mean, variance, and standard deviation of log normalized
gene expression count data in one pass.

Example run::
    python examples/onepass_mean_var_std.py --accelerator gpu --num_workers 4 \
            --default_root_dir runs/onepass

**References:**

    1. *Algorithms for calculating variance*,
       (https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance)
"""

import argparse

import pytorch_lightning as pl
import torch

from scvid.data import (
    DistributedAnnDataCollection,
    IterableDistributedAnnDataCollectionDataset,
    collate_fn,
)
from scvid.module.onepass_mean_var_std import OnePassMeanVarStd
from scvid.train.training_plan import DummyTrainingPlan
from scvid.transforms import ZScoreLog1pNormalize


def main(args):
    # data loader
    dadc = DistributedAnnDataCollection(
        filenames=f"gs://dsp-cell-annotation-service/benchmark_v1/benchmark_v1.{{000..{args.num_shards-1:03}}}.h5ad",
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
        devices=1,
        max_epochs=1,  # one pass
    )
    trainer.fit(plan, train_dataloaders=data_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OnePassMeanVarStd example")
    parser.add_argument(
        "--num_shards", default=325, type=int, help="number of anndata files"
    )
    parser.add_argument("--batch_size", default=10_000, type=int, help="batch size")
    parser.add_argument("--num-workers", default=4, type=int, help="number of workers")
    parser.add_argument(
        "--accelerator", default="cpu", type=str, help="accelerator device"
    )
    args = parser.parse_args()

    main(args)
