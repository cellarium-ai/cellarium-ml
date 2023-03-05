import argparse

import pytorch_lightning as pl
import torch

from scvid.data import (
    DistributedAnnDataCollection,
    DistributedAnnDataCollectionDataset,
    DistributedAnnDataCollectionSingleConsumerSampler,
    collate_fn,
)
from scvid.module.onepass_mean_std import OnePassMeanStd
from scvid.train.training_plan import DummyTrainingPlan
from scvid.transforms import ZScoreLog1pNormalize


def main(args):
    # dataset
    dadc = DistributedAnnDataCollection(
        filenames=f"gs://dsp-cell-annotation-service/benchmark_v1/benchmark_v1.{{000..{args.num_shards-1:03}}}.h5ad",
        shard_size=10_000,
        max_cache_size=1,
    )
    dataset = DistributedAnnDataCollectionDataset(dadc)

    # calculate mean and var
    sampler = DistributedAnnDataCollectionSingleConsumerSampler(
        limits=dadc.limits, shuffle=False
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
    )

    # setup model and training plan
    transform = ZScoreLog1pNormalize(
        mean_g=0, std_g=None, perform_scaling=False, target_count=10_000
    )
    model = OnePassMeanStd(
        transform=transform,
    )
    plan = DummyTrainingPlan(model)

    # train
    device = "gpu" if args.cuda else "cpu"
    trainer = pl.Trainer(
        accelerator=device,
        devices=1,
        max_epochs=1,
    )
    trainer.fit(plan, train_dataloaders=data_loader)

    if args.save is not None:
        stats = {"mean": model.mean, "var": model.var, "std": model.std}
        torch.save(stats, args.save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OnePassMeanVarStd example")
    parser.add_argument(
        "--num-shards", default=50, type=int, help="number of anndata files"
    )
    parser.add_argument("--batch-size", default=5000, type=int, help="batch size")
    parser.add_argument("--num-workers", default=0, type=int, help="number of workers")
    parser.add_argument(
        "--cuda", action="store_true", default=False, help="whether to use cuda"
    )
    parser.add_argument("--save", type=str, help="save parameters to specified file")
    args = parser.parse_args()

    main(args)
