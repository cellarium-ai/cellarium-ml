import math
from typing import Dict

import numpy as np
import torch
from scipy.sparse import issparse
from torch.utils.data import IterableDataset, get_worker_info

from .distributed_anndata import DistributedAnnDataCollection


class DistributedAnnDataCollectionDataset(IterableDataset):
    def __init__(
        self,
        dadc: DistributedAnnDataCollection,
        shuffle: bool,
        seed: int = 0,
        test_mode: bool = False,
    ) -> None:
        self.dadc = dadc
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.test_mode = test_mode

    def __len__(self) -> int:
        return len(self.dadc)

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Return gene counts for a cell at idx."""
        X = self.dadc[idx].X

        data = {}
        data["X"] = X.toarray() if issparse(X) else X

        # for testing purposes
        if self.test_mode:
            worker_info = get_worker_info()
            if worker_info is not None:
                data["worker_id"] = torch.tensor([worker_info.id])
            data["miss_count"] = torch.tensor([self.dadc.cache.miss_count])

        return data

    def __iter__(self):
        if self.test_mode:
            # clear lru cache
            self.dadc.cache.clear()

        if self.shuffle:
            rng = torch.Generator()
            rng.manual_seed(self.seed + self.epoch)
            iter_limits = list(zip([0] + self.dadc.limits, self.dadc.limits))
            # shuffle shards
            limit_indices = torch.randperm(len(iter_limits), generator=rng).tolist()
            cell_idx = []
            for limit_idx in limit_indices:
                lower, upper = iter_limits[limit_idx]
                # shuffle cells within shards
                cell_idx.extend(
                    (torch.randperm(upper - lower, generator=rng) + lower).tolist()
                )
        else:
            cell_idx = list(range(len(self)))

        worker_info = get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = 0
            iter_end = len(self)
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil(len(self) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self))
        yield from (self[cell_idx[i]] for i in range(iter_start, iter_end))
        self.set_epoch(self.epoch + 1)
