from typing import Optional

import pytorch_lightning as pl
import torch

from .dadc_dataset import IterableDistributedAnnDataCollectionDataset
from .distributed_anndata import DistributedAnnDataCollection
from .sampler import collate_fn


class DistributedAnnDataCollectionLoader(pl.LightningDataModule):
    def __init__(
        self,
        dadc: DistributedAnnDataCollection,
        batch_size: Optional[int] = None,
        #  num_replicas: Optional[int] = None,
        #  rank: int = 0,
        num_workers: int = 0,
        shuffle: bool = False,
        seed: int = 0,
        test_mode: bool = False,
    ) -> None:
        self.dadc = dadc
        self.batch_size = batch_size
        #  self.num_replicas = num_replicas
        #  self.rank = rank
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.test_mode = test_mode

    def prepare_data_per_node(self):
        pass

    def train_dataloader(self):
        dataset = IterableDistributedAnnDataCollectionDataset(
            self.dadc,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            test_mode=self.test_mode,
        )
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
        return data_loader
