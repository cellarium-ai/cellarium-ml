import math
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from scipy.sparse import issparse
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .distributed_anndata import DistributedAnnDataCollection


class DistributedAnnDataCollectionDataset(Dataset):
    def __init__(self, dadc: DistributedAnnDataCollection) -> None:
        self.dadc = dadc

    def __len__(self) -> int:
        return len(self.dadc)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Return gene counts for a cell at idx."""
        X = self.dadc[idx].X

        data = {}
        data["X"] = X.toarray() if issparse(X) else X

        return data


class IterableDistributedAnnDataCollectionDataset(IterableDataset):
    r"""
    Iterable DistributedAnnDataCollectionDataset.

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
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
        test_mode: bool = False,
    ) -> None:
        self.dadc = dadc
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
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
        """Return gene counts for cells at idx."""
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

        Example 1:

            indices=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)

            n_obs=12, batch_size=2, num_workers=3
            num_batches=6, batches_per_worker=2, per_worker=2*2=4

                        batch per worker
                          0       1
                      +-----------------+
                    0 | (0,1) | (2,3)   |
                      |-------+---------+
            worker  1 | (4,5) | (6,7)   |
                      +-------+---------+
                    2 | (8,9) | (10,11) |
                      +-------+---------+


        Example 2:

            indices=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

            n_obs=11, batch_size=2, num_workers=2
            num_batches=6, batches_per_worker=3, per_worker=2*3=6

                          batch per worker
                          0       1       2
                      +-----------------------+
                    0 | (0,1) | (2,3) | (4,5) |
            worker    |-------+-------+-------+
                    1 | (6,7) | (8,9) | (10,) |
                      +-------+---------------+


        Example 3:

            indices=(0, 1, 2, 3, 4, 5, 6, 7)

            n_obs=8, batch_size=3, num_workers=2
            num_batches=3, batches_per_worker=2, per_worker=2*3=6

                          batch per worker
                          0       1       2
                      +-------------------+
                    0 | (0,1,2) | (3,4,5) |
            worker    |---------+---------+
                    1 | (6,7)   |         |
                      +---------+---------+
        """
        if self.test_mode:
            # clear lru cache
            self.dadc.cache.clear()

        # gpu nodes
        if not dist.is_available():
            num_replicas = 1
            rank = 0
        else:
            try:
                num_replicas = dist.get_world_size()
                rank = dist.get_rank()
            except RuntimeError:
                num_replicas = 1
                rank = 0
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas-1}]"
            )

        if self.drop_last and len(self) % num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            num_samples = math.ceil((len(self) - num_replicas) / num_replicas)
        else:
            num_samples = math.ceil(len(self) / num_replicas)
        total_size = num_samples * num_replicas

        # workers
        worker_info = get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            worker_id = 0
            num_workers = 1
        else:  # in a worker process
            # split workload
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        num_batches = math.ceil(num_samples / float(self.batch_size))
        batches_per_worker = math.ceil(num_batches / float(num_workers))
        per_worker = batches_per_worker * self.batch_size
        iter_start = worker_id * per_worker + rank * num_samples
        iter_end = min(iter_start + per_worker, (rank + 1) * num_samples)

        # indices
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

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:total_size]
        assert len(indices) == total_size

        yield from (
            self[indices[i : i + self.batch_size]]
            for i in range(iter_start, iter_end, self.batch_size)
        )
        self.set_epoch(self.epoch + 1)
