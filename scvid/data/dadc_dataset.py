# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import Dict, List, Union

import numpy as np
import torch
from scipy.sparse import issparse
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .distributed_anndata import DistributedAnnDataCollection


class DistributedAnnDataCollectionDataset(Dataset):
    r"""
    DistributedAnnDataCollection Dataset.

    Args:
        dadc: DistributedAnnDataCollection from which to load the data.
    """

    def __init__(self, dadc: DistributedAnnDataCollection) -> None:
        self.dadc = dadc

    def __len__(self) -> int:
        return len(self.dadc)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        r"""
        Return feature counts for cells at idx.

        If the count data ``X`` is sparse then it is densified.
        """
        X = self.dadc[idx].X

        data = {}
        data["X"] = X.toarray() if issparse(X) else X

        return data


class IterableDistributedAnnDataCollectionDataset(IterableDataset):
    r"""
    Iterable DistributedAnnDataCollection Dataset.

    Args:
        dadc: DistributedAnnDataCollection from which to load the data.
        batch_size: how many samples per batch to load. Default: ``1``.
        shuffle: set to ``True`` to have the data reshuffled
            at every epoch. Default: ``False``.
        seed: random seed used to shuffle the sampler if :attr:`shuffle=True`. Default: ``0``.
        test_mode: If ``True`` enables tracking of cache and worker informations.
    """

    def __init__(
        self,
        dadc: DistributedAnnDataCollection,
        batch_size: int = 1,
        shuffle: bool = False,
        seed: int = 0,
        test_mode: bool = False,
    ) -> None:
        self.dadc = dadc
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.test_mode = test_mode

    def __len__(self) -> int:
        return len(self.dadc)

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for the iterator. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch.
        """
        self.epoch = epoch

    def __getitem__(self, idx: Union[int, List[int]]) -> Dict[str, np.ndarray]:
        r"""
        Return feature counts for cells at idx.

        If the count data ``X`` is sparse then it is densified.
        """
        X = self.dadc[idx].X

        data = {}
        data["X"] = X.toarray() if issparse(X) else X

        # for testing purposes
        if self.test_mode:
            worker_info = get_worker_info()
            if worker_info is not None:
                data["worker_id"] = np.array([worker_info.id])
            data["miss_count"] = np.array([self.dadc.cache.miss_count])

        return data

    def __iter__(self):
        r"""
        Iterate through the dataset by trying to minimize the amount of anndata files
        fetched by each worker.

        .. note::
            Returned iterator is determined by the ``torch.utils.data.get_worker_info()``
            context. If single worker, then we iterate over entire dataset. If multiple
            workers, we iterate over a subset of cells in a manner that minimizes
            the overlap between the data chunks loaded by each worker.

        Example 1::

            indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            n_obs=12
            batch_size=2
            num_workers=3
            num_batches=6
            batches_per_worker=2
            per_worker=4

        +----------+-------+---------+
        |          |batch 0| batch 1 |
        +==========+=======+=========+
        | worker 0 | (0,1) | (2,3)   |
        +----------+-------+---------+
        | worker 1 | (4,5) | (6,7)   |
        +----------+-------+---------+
        | worker 2 | (8,9) | (10,11) |
        +----------+-------+---------+


        Example 2::

            indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            n_obs=11
            batch_size=2
            num_workers=2
            num_batches=6
            batches_per_worker=3
            per_worker=6

        +----------+-------+-------+-------+
        |          |batch 0|batch 1|batch 2|
        +==========+=======+=======+=======+
        | worker 0 | (0,1) | (2,3) | (4,5) |
        +----------+-------+-------+-------+
        | worker 1 | (6,7) | (8,9) | (10,) |
        +----------+-------+-------+-------+


        Example 3::

            indices=[0, 1, 2, 3, 4, 5, 6, 7]
            n_obs=8
            batch_size=3
            num_workers=2
            num_batches=3
            batches_per_worker=2
            per_worker=6

        +----------+---------+---------+
        |          | batch 0 | batch 1 |
        +==========+=========+=========+
        | worker 0 | (0,1,2) | (3,4,5) |
        +----------+---------+---------+
        | worker 1 | (6,7)   |         |
        +----------+---------+---------+
        """
        if self.test_mode:
            # clear lru cache
            self.dadc.cache.clear()

        if self.shuffle:
            rng = torch.Generator()
            rng.manual_seed(self.seed + self.epoch)
            iter_limits = list(zip([0] + self.dadc.limits, self.dadc.limits))
            # shuffle shards
            limit_indices = torch.randperm(len(iter_limits), generator=rng).tolist()
            indices = []
            for limit_idx in limit_indices:
                lower, upper = iter_limits[limit_idx]
                # shuffle cells within shards
                indices.extend(
                    (torch.randperm(upper - lower, generator=rng) + lower).tolist()
                )
        else:
            indices = list(range(len(self)))

        worker_info = get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = 0
            iter_end = len(self)
        else:  # in a worker process
            # split workload
            num_batches = int(math.ceil(len(self) / float(self.batch_size)))
            batches_per_worker = int(
                math.ceil(num_batches / float(worker_info.num_workers))
            )
            per_worker = batches_per_worker * self.batch_size
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self))

        yield from (
            self[indices[i : i + self.batch_size]]
            for i in range(iter_start, iter_end, self.batch_size)
        )
        self.set_epoch(self.epoch + 1)
