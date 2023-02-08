from typing import Iterator, Optional, TypeVar

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from . import DistributedAnnDataCollectionDataset

T_co = TypeVar("T_co", covariant=True)

#
# TODO -- rather than take `shard_size` as a constructor parameter, it should get and 
# make use of the `limits` directly from the DistributedAnnDataCollection object
class DistributedAnnDataCollectionSampler(DistributedSampler):
    def __init__(
        self,
        dataset: DistributedAnnDataCollectionDataset,
        shard_size: int,
        seed: int = 0,
        shuffle: bool = True,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval" " [0, {}]".format(rank, num_replicas - 1)
            )

        self.dataset = dataset
        self.shard_size = shard_size
        self.seed = seed
        self.shuffle = shuffle
        self.num_replicas = num_replicas
        self.rank = rank

        self.set_epoch(0)

    def __iter__(self) -> Iterator[T_co]:

        # (3) iterate through chunks, shuffling within each chunk if desired
        # TODO: shuffle across a set of N shards at a time
        for si in self.process_shard_indexes:
            offset = si * self.shard_size

            # what is most efficient way to do this permutation?
            for i in torch.randperm(self.shard_size, generator=self.g).tolist():
                yield offset + i

    def __len__(self) -> int:
        return (self.process_shard_indexes) * self.shard_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

        # Sampling Procedure
        #   - (1) Use seed and epoch to generate a random order of shards (divisible by world_size)
        #   - (2) Randomly evenly partition the global list by global_rank (a per process list)
        print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
        print(f"DEBUG: using rank {self.rank}")
        print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")

        num_shards = len(self.dataset.dac.filenames)
        capped_shards = (num_shards // (self.num_replicas)) * (self.num_replicas)

        self.g = torch.Generator()
        self.g.manual_seed(self.seed + self.epoch)

        # NOTE: how do we ensure the last shard is complete (same size)
        all_shards_indexes = torch.randperm(num_shards, generator=self.g).tolist()[0:capped_shards]
        print(f"All Shards: {all_shards_indexes}")

        # (2) partition by process (rank)
        self.process_shard_indexes = [
            all_shards_indexes[i] for i in range(len(all_shards_indexes)) if (i % self.num_replicas) == self.rank
        ]
        print(f"Rank: {self.rank} Process Chunks: {self.process_shard_indexes}")
