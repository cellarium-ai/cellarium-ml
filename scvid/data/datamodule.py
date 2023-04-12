# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Iterable, Sequence

import lightning.pytorch as pl
import torch
from anndata.experimental.multi_files._anncollection import ConvertType

from .dadc_dataset import IterableDistributedAnnDataCollectionDataset
from .distributed_anndata import DistributedAnnDataCollection
from .util import collate_fn


class DistributedAnnDataCollectionDataModule(pl.LightningDataModule):
    """
    DataModule for DistributedAnnDataCollection and IterableDistributedAnnDataCollectionDataset.
    """

    def __init__(
        self,
        # DistributedAnnDataCollection args
        filenames: Sequence[str] | str,
        limits: Iterable[int] | None = None,
        shard_size: int | None = None,
        last_shard_size: int | None = None,
        max_cache_size: int = 1,
        cache_size_strictly_enforced: bool = True,
        label: str | None = None,
        keys: Sequence[str] | None = None,
        index_unique: str | None = None,
        convert: ConvertType | None = None,
        indices_strict: bool = True,
        # IterableDistributedAnnDataCollectionDataset args
        batch_size: int = 1,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
        test_mode: bool = False,
        # DataLoader args
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        # DistributedAnnDataCollection args
        self.filenames = filenames
        self.limits = limits
        self.shard_size = shard_size
        self.last_shard_size = last_shard_size
        self.max_cache_size = max_cache_size
        self.cache_size_strictly_enforced = cache_size_strictly_enforced
        self.label = label
        self.keys = keys
        self.index_unique = index_unique
        self.convert = convert
        self.indices_strict = indices_strict
        # IterableDistributedAnnDataCollectionDataset args
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.test_mode = test_mode
        # DataLoader args
        self.num_workers = num_workers

    @property
    def n_obs(self) -> int:
        return self.dadc.n_obs

    @property
    def n_vars(self) -> int:
        return self.dadc.n_vars

    def setup(self, stage: str | None = None) -> None:
        """
        .. note::
           setup is called from every process across all the nodes. Setting state here is recommended.
        """
        self.dadc = DistributedAnnDataCollection(
            filenames=self.filenames,
            limits=self.limits,
            shard_size=self.shard_size,
            last_shard_size=self.last_shard_size,
            max_cache_size=self.max_cache_size,
            cache_size_strictly_enforced=self.cache_size_strictly_enforced,
            label=self.label,
            keys=self.keys,
            index_unique=self.index_unique,
            convert=self.convert,
            indices_strict=self.indices_strict,
        )
        self.dataset = IterableDistributedAnnDataCollectionDataset(
            dadc=self.dadc,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            seed=self.seed,
            drop_last=self.drop_last,
            test_mode=self.test_mode,
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.dataset,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
