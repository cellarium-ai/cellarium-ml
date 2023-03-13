import math
from typing import Dict, List, Union

import numpy as np
import torch
from scipy.sparse import issparse
from torch.utils.data import Dataset, IterableDataset

from .distributed_anndata import DistributedAnnDataCollection
from .util import get_rank_and_num_replicas, get_worker_info


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
        drop_last: If ``True``, then the sampler will drop the tail of the data
            to make it evenly divisible across the number of replicas. If ``False``,
            the sampler will add extra indices to make the data evenly divisible across
            the replicas. Default: ``False``.
        test_mode: If ``True`` enables tracking of cache and worker informations.
    """

    def __init__(
        self,
        dadc: DistributedAnnDataCollection,
        batch_size: int = 1,
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
        r"""
        Return feature counts for cells at idx.

        If the count data ``X`` is sparse then it is densified.
        """
        X = self.dadc[idx].X

        data = {}
        data["X"] = X.toarray() if issparse(X) else X

        # for testing purposes
        if self.test_mode:
            rank, num_replicas = get_rank_and_num_replicas()
            worker_id, num_workers = get_worker_info()
            data["rank"] = np.array([rank])
            data["num_replicas"] = np.array([num_replicas])
            data["worker_id"] = np.array([worker_id])
            data["num_workers"] = np.array([num_workers])
            data["miss_count"] = np.array([self.dadc.cache.miss_count])

        return data

    def __iter__(self):
        r"""
        Iterate through the dataset by trying to minimize the amount of anndata files
        fetched by each worker.

        .. note::
            Returned iterator is determined by the ``torch.utils.data.get_worker_info()``
            and ``torch.distributed`` contexts. Indices are evenly divided between replicas
            (see :attr:`drop_last`). If single worker, then we iterate over entire replica.
            If multiple workers, we iterate over a subset of replica in a manner that minimizes
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

        # replicas
        rank, num_replicas = get_rank_and_num_replicas()

        if self.drop_last and len(self) % num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data.
            per_replica = len(self) // num_replicas
        else:
            per_replica = math.ceil(len(self) / num_replicas)
        total_size = per_replica * num_replicas
        batches_per_replica = math.ceil(per_replica / float(self.batch_size))

        # workers
        worker_id, num_workers = get_worker_info()

        batches_per_worker = math.ceil(batches_per_replica / float(num_workers))
        per_worker = batches_per_worker * self.batch_size

        # split workload
        iter_start = worker_id * per_worker
        iter_end = min(iter_start + per_worker, per_replica)

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
        indices = indices[rank * per_replica : (rank + 1) * per_replica]
        assert len(indices) == per_replica

        yield from (
            self[indices[i : i + self.batch_size]]
            for i in range(iter_start, iter_end, self.batch_size)
        )
        self.set_epoch(self.epoch + 1)
