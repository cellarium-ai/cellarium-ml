from typing import Iterator, Sequence

import torch
from torch.utils.data.sampler import Sampler


class DADCSampler(Sampler):
    """
    Sampler for DistributedAnnDataCollection.

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

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            rng = torch.Generator()
            rng.manual_seed(self.seed)
            iter_limits = list(zip([0] + self.limits, self.limits))
            limit_indices = torch.randperm(len(iter_limits), generator=rng).tolist()
            for limit_idx in limit_indices:
                lower, upper = iter_limits[limit_idx]
                yield from (torch.randperm(upper - lower) + lower).tolist()
        else:
            yield from range(self.n_obs)

    def __len__(self):
        return self.n_obs


def collate_fn(batch):
    return {"X": torch.cat([torch.from_numpy(data["X"]) for data in batch], 0)}
