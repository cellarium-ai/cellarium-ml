# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math

import numpy as np
import torch
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
        ...         "feature_g": AnnDataField(attr="var_names"),
        ...     },
        ...     batch_size=5000,
        ...     shuffle=True,
        ...     seed=0,
        ...     drop_last=True,
        ... )

    Args:
        dadc:
            DistributedAnnDataCollection from which to load the data.
        batch_keys:
            Dictionary that specifies which attributes and keys of the :attr:`dadc` to return
            in the batch data and how to convert them. Keys must correspond to
            the input keys of the transforms or the model. Values must be instances of
            :class:`cellarium.ml.utilities.data.AnnDataField`.
        batch_size:
            How many samples per batch to load.
        shuffle:
            If ``True``, the data is reshuffled at every epoch.
        seed:
            Random seed used to shuffle the sampler if :attr:`shuffle=True`.
        drop_last:
            If ``True``, then the sampler will drop the tail of the data
            to make it evenly divisible across the number of replicas. If ``False``,
            the sampler will add extra indices to make the data evenly divisible across
            the replicas.
        test_mode:
            If ``True``, then tracking of cache and worker informations will be enabled.
    """

    def __init__(
        self,
        dadc: DistributedAnnDataCollection,
        batch_keys: dict[str, AnnDataField],
        batch_size: int = 1,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
        test_mode: bool = False,
    ) -> None:
        self.dadc = dadc
        self.batch_keys = batch_keys
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.test_mode = test_mode

    def __len__(self) -> int:
        """
        Returns the number of batches per replica.
        """
        _, num_replicas = get_rank_and_num_replicas()

        if self.drop_last and len(self.dadc) % num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data.
            per_replica = len(self.dadc) // num_replicas
        else:
            per_replica = math.ceil(len(self.dadc) / num_replicas)
        return math.ceil(per_replica / float(self.batch_size))

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for the iterator. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch.
        """
        self.epoch = epoch

    def __getitem__(self, idx: int | list[int] | slice) -> dict[str, np.ndarray]:
        r"""
        Returns a dictionary containing the data from the :attr:`dadc` with keys specified by its :attr:`dadc.convert`
        at the given index ``idx``.
        """

        data = {}
        for key, field in self.batch_keys.items():
            data[key] = field(self.dadc)[idx]

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
        Iterate through the dataset by trying to minimize the amount of anndata files
        fetched by each worker.

        .. note::
            Returned iterator is determined by the ``torch.utils.data.get_worker_info()``
            and ``torch.distributed`` contexts. Iterated indices are evenly divided between replicas
            (see :attr:`drop_last`). If multiple workers per replica, then indices are further
            divided between workers (last worker might contain less indices than other workers, see
            examples below). Indices are shuffled and iterated in a manner that minimizes the overlap
            between the data chunks loaded by each worker.

        Example 1::

            indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            n_obs=12
            num_replicas=1
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
            num_replicas=1
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
            num_replicas=1
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


        Example 4::

            indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            n_obs=11
            num_replicas=2
            drop_last=True

            truncated_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            total_size=10

            first_replica=[0, 1, 2, 3, 4]
            batch_size=2
            num_workers=1
            num_batches=3
            batches_per_worker=3
            per_worker=6

            second_replica=[5, 6, 7, 8, 9]
            batch_size=2
            num_workers=1
            num_batches=3
            batches_per_worker=3
            per_worker=6

        *Replica 1*

        +----------+-------+-------+-------+
        |          |batch 0|batch 1|batch 2|
        +==========+=======+=======+=======+
        | worker 0 | (0,1) | (2,3) | (4,)  |
        +----------+-------+-------+-------+

        *Replica 2*

        +----------+-------+-------+-------+
        |          |batch 0|batch 1|batch 2|
        +==========+=======+=======+=======+
        | worker 0 | (5,6) | (7,8) | (9,)  |
        +----------+-------+-------+-------+


        Example 5::

            indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            n_obs=11
            num_replicas=2
            drop_last=False

            padded_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0]
            total_size=12

            first_replica=[0, 1, 2, 3, 4, 5]
            batch_size=2
            num_workers=1
            num_batches=3
            batches_per_worker=3
            per_worker=6

            second_replica=[6, 7, 8, 9, 10, 0]
            batch_size=2
            num_workers=1
            num_batches=3
            batches_per_worker=3
            per_worker=6

        *Replica 1*

        +----------+-------+-------+-------+
        |          |batch 0|batch 1|batch 2|
        +==========+=======+=======+=======+
        | worker 0 | (0,1) | (2,3) | (4,5) |
        +----------+-------+-------+-------+

        *Replica 2*

        +----------+-------+-------+--------+
        |          |batch 0|batch 1|batch 2 |
        +==========+=======+=======+========+
        | worker 0 | (6,7) | (8,9) | (10,0) |
        +----------+-------+-------+--------+
        """
        if self.test_mode:
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
                indices.extend((torch.randperm(upper - lower, generator=rng) + lower).tolist())
        else:
            indices = list(range(len(self.dadc)))

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
        indices = indices[rank * per_replica : (rank + 1) * per_replica]
        assert len(indices) == per_replica

        yield from (self[indices[i : i + self.batch_size]] for i in range(iter_start, iter_end, self.batch_size))
        # Sets epoch for persistent workers
        self.set_epoch(self.epoch + 1)
