import random

import torch
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from torch import nn, optim
from torch.distributed import get_rank, get_world_size
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from scvi.data import LazyAnnData, read_h5ad_gcs


# define the LightningModule
class LitAutoEncoder(LightningModule):
    def __init__(self):
        super().__init__()
        features = 36350
        self.encoder = nn.Sequential(nn.Linear(features, 64), nn.ReLU(), nn.Linear(64, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, features))

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class DataParallelIterableLazyAnnDataDataset(IterableDataset):
    def __init__(self, ladata, world_size: int, global_rank: int, epoch: int = 0, seed: int = 0):
        super(DataParallelIterableLazyAnnDataDataset).__init__()
        self.ladata = ladata
        self.shard_names = ladata.shard_names
        self.seed = seed
        self.world_size = world_size
        self.global_rank = global_rank
        self.epoch = epoch

    # see https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        #
        # Sampling Procedure
        #   - (1) Use seed and epoch to generate a random order of shards (divisible by world_size)
        #   - (2) Randomly evenly partition the global list by global_rank (a per process list)
        #   - (3) Randomly evenly partition the process list by worker_id
        #   - (4) Iterate through data
        #
        # Handles multi-node, multi-process, multi-worker dataloading efficiently without duplicating
        print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
        print(f"DEBUG: using rank {self.global_rank} and worker_info {get_worker_info()}")
        print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")

        # (1) generate fill randomized list of shards evenly divisible by world_size * num_workers
        worker_info = get_worker_info()
        if worker_info is None:
            num_workers = 1
        else:
            num_workers = worker_info.num_workers

        num_shards = len(self.shard_names)
        capped_shards = (num_shards // (self.world_size * num_workers)) * (self.world_size * num_workers)

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        all_shards_indexes = torch.randperm(num_shards, generator=g).tolist()[0:capped_shards]

        print(f"All Shards: {all_shards_indexes}")

        # (2) partition by process (rank)
        process_shard_indexes = [
            all_shards_indexes[i] for i in range(len(all_shards_indexes)) if (i % self.world_size) == self.global_rank
        ]

        print(f"Rank: {self.global_rank} Process Chunks: {process_shard_indexes}")

        # (3) then divide up by workers (may be None, implies one)

        if worker_info is None:  # single-process data loading, return the full iterator
            worker_shard_indexes = process_shard_indexes
            print(f"Global Rank {self.global_rank} and no defined workers got {worker_shard_indexes}")
        else:  # in a worker process
            print(worker_info)
            # split workload
            worker_shard_indexes = [
                process_shard_indexes[i]
                for i in range(len(process_shard_indexes))
                if (i % worker_info.num_workers) == worker_info.id
            ]
            print(f"Global Rank {self.global_rank} and worker {worker_info.id} got {worker_shard_indexes}")

        # (4) iterate through chunks (localize, load, shuffle, yield)
        for i in worker_shard_indexes:
            adata = read_h5ad_gcs(self.shard_names[i])

            # borrowed from Fedor's dataset.py
            data = torch.Tensor(adata.raw.X.toarray().astype(int))
            db_ids = torch.Tensor(adata.obs_names.values.astype(int))

            # TODO smarter shuffle -- see https://discuss.pytorch.org/t/shuffling-a-tensor/25422
            temp = list(zip(data, db_ids))
            random.shuffle(temp)
            for d in temp:
                yield d

            # attempt to free memory
            del adata
            del temp
            del data
            del db_ids


class CustomDataModule(LightningDataModule):
    def __init__(self, url_pattern: str, batch_size: int = 4):
        super().__init__()
        self.ladata = LazyAnnData(url_pattern, shard_size=10000)
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.world_size = get_world_size()
        self.global_rank = get_rank()
        print(f"Initializing Data Module with world size {self.world_size} and process rank {self.global_rank}")

    def train_dataloader(self):
        train_dataset = DataParallelIterableLazyAnnDataDataset(self.ladata, self.world_size, self.global_rank)
        return DataLoader(
            train_dataset, batch_size=self.batch_size, num_workers=2, persistent_workers=True, shuffle=False
        )


def main():
    model = LitAutoEncoder()
    url_pattern = "gs://dsp-cell-annotation-service/benchmark_v1/benchmark_v1.{000..018}.h5ad"
    dm = CustomDataModule(url_pattern, batch_size=100)

    trainer = Trainer(accelerator="gpu", devices=2, replace_sampler_ddp=False)
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
