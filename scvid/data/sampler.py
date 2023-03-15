# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Iterator, Sequence

import torch
from torch.utils.data.sampler import Sampler


class DistributedAnnDataCollectionSingleConsumerSampler(Sampler):
    """
    Single consumer sampler for DistributedAnnDataCollection.

    Args:
        limits: Limits of cell indices for anndata files.
        shuffle: If ``True``, sampler will shuffle the indices.
        seed: random seed used to shuffle the sampler if :attr:`shuffle=True`. Default: ``0``.
    """

    def __init__(
        self,
        limits: Sequence[int],
        shuffle: bool,
        seed: int = 0,
    ) -> None:
        self.limits = list(limits)
        self.n_obs = self.limits[-1]
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        r"""
        Shuffling is performed by first shuffling shards and then shuffling cells within shards.
        """
        if self.shuffle:
            rng = torch.Generator()
            rng.manual_seed(self.seed + self.epoch)
            iter_limits = list(zip([0] + self.limits, self.limits))
            # shuffle shards
            limit_indices = torch.randperm(len(iter_limits), generator=rng).tolist()
            for limit_idx in limit_indices:
                lower, upper = iter_limits[limit_idx]
                # shuffle cells within shards
                yield from (
                    torch.randperm(upper - lower, generator=rng) + lower
                ).tolist()
        else:
            yield from range(self.n_obs)
        self.set_epoch(self.epoch + 1)

    def __len__(self):
        return self.n_obs

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


def collate_fn(batch):
    keys = batch[0].keys()
    return {
        key: torch.cat([torch.from_numpy(data[key]) for data in batch], dim=0)
        for key in keys
    }
