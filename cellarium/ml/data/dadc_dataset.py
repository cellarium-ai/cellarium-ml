# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math
from itertools import islice
from typing import Literal

import numpy as np
import torch
from anndata import AnnData
from boltons.iterutils import chunked_iter
from torch.utils.data import IterableDataset

from cellarium.ml.data.distributed_anndata import DistributedAnnDataCollection
from cellarium.ml.utilities.data import AnnDataField, get_rank_and_num_replicas, get_worker_info


class IterableDistributedAnnDataCollectionDataset(IterableDataset):
    r"""
    Iterable DistributedAnnDataCollection Dataset.

    When :attr:`shuffle` is set to ``True`` then the iterator yields datapoints that are
    uniformly sampled from the entire dataset. Typical use cases include training variational
    models using the stochastic gradient descent algorithm.

    In order to maximize buffer usage, we only shuffle shards and datapoints within individual
    shards (and not across shards). Therefore, to achieve unbiased pseudo-random uniform sampling,
    it is imperative that the shards themselves contain datapoints that are uniformly sampled
    from the entire dataset. If correlations exist between datapoints in a given shard (e.g. all
    cells coming from the same tissue or experiment), then this assumption is violated. It is
    the user's responsibility to prepare appropriately shuffled data shards.

    Example::

        >>> from cellarium.ml.data import (
        ...     DistributedAnnDataCollection,
        ...     IterableDistributedAnnDataCollectionDataset,
        ... )
        >>> from cellarium.ml.utilities.data import AnnDataField, densify

        >>> dadc = DistributedAnnDataCollection(
        ...     "gs://bucket-name/folder/adata{000..005}.h5ad",
        ...     shard_size=10_000,
        ...     max_cache_size=2)

        >>> dataset = IterableDistributedAnnDataCollectionDataset(
        ...     dadc,
        ...     batch_keys={
        ...         "x_ng": AnnDataField(attr="X", convert_fn=densify),
        ...         "var_names_g": AnnDataField(attr="var_names"),
        ...     },
        ...     batch_size=5000,
        ...     iteration_strategy="cache_efficient",
        ...     shuffle=True,
        ...     seed=0,
        ...     drop_last=True,
        ... )

    Args:
        dadc:
            DistributedAnnDataCollection or AnnData from which to load the data.
        batch_keys:
            Dictionary that specifies which attributes and keys of the :attr:`dadc` to return
            in the batch data and how to convert them. Keys must correspond to
            the input keys of the transforms or the model. Values must be instances of
            :class:`cellarium.ml.utilities.data.AnnDataField`.
        batch_size:
            How many samples per batch to load.
        iteration_strategy:
            Strategy to use for iterating through the dataset. Options are ``same_order`` and ``cache_efficient``.
            ``same_order`` will iterate through the dataset in the same order independent of the number of replicas
            and workers. ``cache_efficient`` will try to minimize the amount of anndata files fetched by each worker.
        shuffle:
            If ``True``, the data is reshuffled at every epoch.
        seed:
            Random seed used to shuffle the sampler if :attr:`shuffle=True`.
        drop_last:
            If ``True``, then the sampler will drop the tail of the data
            to make it evenly divisible across the number of replicas. If ``False``,
            the sampler will add extra indices to make the data evenly divisible across
            the replicas.
        start_idx:
            The starting index of the dataset. If ``None``, then the dataset will start from the first index.
        end_idx:
            The ending index (exclusive) of the dataset. If ``None``, then the dataset will end at
            the last index (inclusive).
        test_mode:
            If ``True``, then tracking of cache and worker informations will be enabled.
    """

    def __init__(
        self,
        dadc: DistributedAnnDataCollection | AnnData,
        batch_keys: dict[str, AnnDataField],
        batch_size: int = 1,
        iteration_strategy: Literal["same_order", "cache_efficient"] = "cache_efficient",
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
        start_idx: int | None = None,
        end_idx: int | None = None,
        test_mode: bool = False,
    ) -> None:
        self.dadc = dadc
        if isinstance(dadc, AnnData):
            # mimic a DistributedAnnDataCollection
            self.dadc.limits = [dadc.n_obs]
        self.batch_keys = batch_keys
        self.batch_size = batch_size
        self.iteration_strategy = iteration_strategy
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.start_idx = 0 if start_idx is None else start_idx
        self.end_idx = dadc.n_obs if end_idx is None else end_idx
        self.epoch = 0
        self.test_mode = test_mode

    def __len__(self) -> int:
        """
        Returns the number of batches per replica.
        """
        _, num_replicas = get_rank_and_num_replicas()

        n_obs = self.end_idx - self.start_idx
        if self.drop_last and n_obs % num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data.
            per_replica = n_obs // num_replicas
        else:
            per_replica = math.ceil(n_obs / num_replicas)
        return math.ceil(per_replica / float(self.batch_size))

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for the iterator. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch.
        """
        self.epoch = epoch

    def __getitem__(self, idx: int | list[int] | slice) -> dict[str, np.ndarray]:
        r"""
        Returns a dictionary containing the data from the :attr:`dadc` with keys specified by the :attr:`batch_keys`
        at the given index ``idx``.
        """

        data = {}
        for key, field in self.batch_keys.items():
            data[key] = field(self.dadc, idx)

        # for testing purposes
        if self.test_mode:
            rank, num_replicas = get_rank_and_num_replicas()
            worker_id, num_workers = get_worker_info()
            data["rank"] = np.array([rank])
            data["num_replicas"] = np.array([num_replicas])
            data["worker_id"] = np.array([worker_id])
            data["num_workers"] = np.array([num_workers])
            data["miss_count"] = np.array([self.dadc.cache.miss_count])
            data["epoch"] = np.array([self.epoch])

        return data

    def __iter__(self):
        r"""
        Iterate through the dataset by trying to minimize the amount of anndata files fetched by each worker.
        Iterated indices are evenly divided between replicas (see :attr:`drop_last`).

        .. note::

            1. For both strategies the amount of anndata files fetched is reduced by
               shuffling the shards first and then the datapoints within the shards.
            2. ``same_order`` strategy will iterate through the dataset in the same order independent
               of the number of replicas and workers.
            3. For ``cache_efficient`` strategy the amount of anndata files fetched is further
               reduced by assigning to each worker a contiguous chunk of the dataset.
               The returned iterator is determined by the ``torch.utils.data.get_worker_info()``
               and ``torch.distributed`` contexts.

        **Example 1**::

            indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            num_replicas=1
            batch_size=2
            num_workers=3

        Same order:

        +------------+-------+-------+-------+-------+-------+---------+
        | batch idx  | 0     | 1     | 2     | 3     | 4     | 5       |
        +============+=======+=======+=======+=======+=======+=========+
        | indices    | (0,1) | (2,3) | (4,5) | (6,7) | (8,9) | (10,11) |
        +------------+-------+-------+-------+-------+-------+---------+
        | worker id  | 0     | 1     | 2     | 0     | 1     | 2       |
        +------------+-------+-------+-------+-------+-------+---------+

        Cache efficient:

        +------------+-------+-------+-------+-------+-------+---------+
        | batch idx  | 0     | 1     | 2     | 3     | 4     | 5       |
        +============+=======+=======+=======+=======+=======+=========+
        | indices    | (0,1) | (4,5) | (8,9) | (2,3) | (6,7) | (10,11) |
        +------------+-------+-------+-------+-------+-------+---------+
        | worker id  | 0     | 1     | 2     | 0     | 1     | 2       |
        +------------+-------+-------+-------+-------+-------+---------+


        **Example 2**::

            indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            num_replicas=1
            batch_size=2
            num_workers=2

        Same order:

        +------------+-------+-------+-------+-------+-------+---------+
        | batch idx  | 0     | 1     | 2     | 3     | 4     | 5       |
        +============+=======+=======+=======+=======+=======+=========+
        | indices    | (0,1) | (2,3) | (4,5) | (6,7) | (8,9) | (10,)   |
        +------------+-------+-------+-------+-------+-------+---------+
        | worker id  | 0     | 1     | 0     | 1     | 0     | 1       |
        +------------+-------+-------+-------+-------+-------+---------+

        Cache efficient:

        +------------+-------+-------+-------+-------+-------+---------+
        | batch idx  | 0     | 1     | 2     | 3     | 4     | 5       |
        +============+=======+=======+=======+=======+=======+=========+
        | indices    | (0,1) | (6,7) | (2,3) | (8,9) | (4,5) | (10,)   |
        +------------+-------+-------+-------+-------+-------+---------+
        | worker id  | 0     | 1     | 0     | 1     | 0     | 1       |
        +------------+-------+-------+-------+-------+-------+---------+

        **Example 3**::

            indices=[0, 1, 2, 3, 4, 5, 6, 7]
            num_replicas=1
            batch_size=3
            num_workers=2

        Same order:

        +------------+---------+---------+-------+
        | batch idx  | 0       | 1       | 2     |
        +============+=========+=========+=======+
        | indices    | (0,1,2) | (3,4,5) | (6,7) |
        +------------+---------+---------+-------+
        | worker id  | 0       | 1       | 0     |
        +------------+---------+---------+-------+

        Cache efficient:

        +------------+---------+-------+---------+
        | batch idx  | 0       | 1     | 2       |
        +============+=========+=======+=========+
        | indices    | (0,1,2) | (6,7) | (3,4,5) |
        +------------+---------+-------+---------+
        | worker id  | 0       | 1     | 0       |
        +------------+---------+-------+---------+

        **Example 4**::

            indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            num_replicas=2
            drop_last=True
            batch_size=2
            num_workers=1

        Same order:

        *Replica 1*

        +------------+-------+-------+------+
        | batch idx  | 0     | 1     | 2    |
        +============+=======+=======+======+
        | indices    | (0,2) | (4,6) | (8,) |
        +------------+-------+-------+------+
        | worker id  | 0     | 0     | 0    |
        +------------+-------+-------+------+

        *Replica 2*

        +------------+-------+-------+------+
        | batch idx  | 0     | 1     | 2    |
        +============+=======+=======+======+
        | indices    | (1,3) | (5,7) | (9,) |
        +------------+-------+-------+------+
        | worker id  | 0     | 0     | 0    |
        +------------+-------+-------+------+

        Cache efficient:

        *Replica 1*

        +------------+-------+-------+------+
        | batch idx  | 0     | 1     | 2    |
        +============+=======+=======+======+
        | indices    | (0,1) | (2,3) | (4,) |
        +------------+-------+-------+------+
        | worker id  | 0     | 0     | 0    |
        +------------+-------+-------+------+

        *Replica 2*

        +------------+-------+-------+------+
        | batch idx  | 0     | 1     | 2    |
        +============+=======+=======+======+
        | indices    | (5,6) | (7,8) | (9,) |
        +------------+-------+-------+------+
        | worker id  | 0     | 0     | 0    |
        +------------+-------+-------+------+


        **Example 5**::

            indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            num_replicas=2
            drop_last=False
            batch_size=2
            num_workers=1

        Same order:

        *Replica 1*

        +------------+-------+-------+--------+
        | batch idx  | 0     | 1     | 2      |
        +============+=======+=======+========+
        | indices    | (0,2) | (4,6) | (8,10) |
        +------------+-------+-------+--------+
        | worker id  | 0     | 0     | 0      |
        +------------+-------+-------+--------+

        *Replica 2*

        +------------+-------+-------+-------+
        | batch idx  | 0     | 1     | 2     |
        +============+=======+=======+=======+
        | indices    | (1,3) | (5,7) | (9,0) |
        +------------+-------+-------+-------+
        | worker id  | 0     | 0     | 0     |
        +------------+-------+-------+-------+

        Cache efficient:

        *Replica 1*

        +------------+-------+-------+-------+
        | batch idx  | 0     | 1     | 2     |
        +============+=======+=======+=======+
        | indices    | (0,1) | (2,3) | (4,5) |
        +------------+-------+-------+-------+
        | worker id  | 0     | 0     | 0     |
        +------------+-------+-------+-------+

        *Replica 2*

        +------------+-------+-------+--------+
        | batch idx  | 0     | 1     | 2      |
        +============+=======+=======+========+
        | indices    | (6,7) | (8,9) | (10,0) |
        +------------+-------+-------+--------+
        | worker id  | 0     | 0     | 0      |
        +------------+-------+-------+--------+
        """
        if self.test_mode and isinstance(self.dadc, DistributedAnnDataCollection):
            # clear lru cache
            self.dadc.cache.clear()

        # replicas
        rank, num_replicas = get_rank_and_num_replicas()

        if self.drop_last and len(self.dadc) % num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data.
            per_replica = len(self.dadc) // num_replicas
        else:
            per_replica = math.ceil(len(self.dadc) / num_replicas)
        total_size = per_replica * num_replicas

        # workers
        worker_id, num_workers = get_worker_info()

        # indices
        if self.shuffle:
            rng = torch.Generator()
            rng.manual_seed(self.seed + self.epoch)
            limits = [idx for idx in self.dadc.limits if idx > self.start_idx and idx < self.end_idx]
            iter_limits = list(zip([self.start_idx] + limits, limits + [self.end_idx]))
            # shuffle shards
            limit_indices = torch.randperm(len(iter_limits), generator=rng).tolist()
            indices = []
            for limit_idx in limit_indices:
                lower, upper = iter_limits[limit_idx]
                # shuffle cells within shards
                indices.extend((torch.randperm(upper - lower, generator=rng) + lower).tolist())
        else:
            indices = list(range(self.start_idx, self.end_idx))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:total_size]

        if self.iteration_strategy == "same_order":
            # replica indices
            indices = indices[rank:total_size:num_replicas]
            if len(indices) != per_replica:
                raise ValueError(
                    f"The number of indices must be equal to the per_replica size. "
                    f"Got {len(indices)} != {per_replica} at rank {rank}."
                )

            # in python 3.12 `chunked_iter` can be replaced with `itertools.batched`
            for batch_indices in islice(chunked_iter(indices, self.batch_size), worker_id, None, num_workers):
                yield self[batch_indices]

        elif self.iteration_strategy == "cache_efficient":
            # replica indices
            indices = indices[rank * per_replica : (rank + 1) * per_replica]
            if len(indices) != per_replica:
                raise ValueError(
                    f"The number of indices must be equal to the per_replica size. "
                    f"Got {len(indices)} != {per_replica} at rank {rank}."
                )

            # worker indices
            batches_per_replica = math.ceil(per_replica / float(self.batch_size))
            batches_per_worker = math.ceil(batches_per_replica / float(num_workers))
            per_worker = batches_per_worker * self.batch_size

            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, per_replica)
            indices = indices[iter_start:iter_end]

            # in python 3.12 `chunked_iter` can be replaced with `itertools.batched`
            for batch_indices in chunked_iter(indices, self.batch_size):
                yield self[batch_indices]

        # Sets epoch for persistent workers
        self.set_epoch(self.epoch + 1)
