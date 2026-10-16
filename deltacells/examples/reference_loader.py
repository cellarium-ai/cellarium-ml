# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
A reference ``IterableDataset`` showing how to drive a :class:`deltacells.DeltaCellsDataset` from a PyTorch ``DataLoader``.

This is an *example*, not part of the package API: cellarium-ml's own orchestrator (shuffling, replica / worker splitting, resume,
``drop_last_indices`` ...) would call the same three methods -- ``prefetch`` / ``batch_nnz`` / ``get_batch`` -- in the same way.

The iteration order mirrors the "cache efficient" strategy of the h5ad loader: each epoch the tiles are shuffled, each
(replica, worker) takes a contiguous run of that shuffled tile list, cells are shuffled within each tile, and batches are cut
from the resulting stream (so a batch occasionally spans the boundary between two tiles). While a worker processes tile ``k`` the
next few tiles are already being fetched.
"""

from __future__ import annotations

import numpy as np
import torch

from deltacells import DeltaCellsDataset


class TileShuffleDataset(torch.utils.data.IterableDataset):
    """Yields dicts ``{"indptr", "indices", "values"}`` of CPU tensors (CSR of one batch; ``values`` float32).

    Args:
        dataset: The dataset to read (pickled to the DataLoader workers; caches are per worker).
        batch_size: Cells per batch.
        shuffle: Shuffle tiles (per epoch) and cells (within tiles).
        seed: Seed; the order depends on ``seed + epoch`` only, so it is reproducible and independent of the number of workers
            *within* each worker's share.
        drop_last: Drop the final partial batch of each worker.
        prefetch_tiles: How many upcoming tiles to ask the dataset to fetch ahead.
        shared_memory: Allocate the output tensors in shared memory (what a worker should do), avoiding a copy when the batch is
            sent to the main process.
        num_replicas, rank: Data-parallel split (default: taken from ``torch.distributed`` if initialized).
    """

    def __init__(
        self,
        dataset: DeltaCellsDataset,
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        prefetch_tiles: int = 2,
        shared_memory: bool = True,
        num_replicas: int | None = None,
        rank: int | None = None,
    ) -> None:
        self.dataset, self.batch_size, self.shuffle, self.seed = dataset, batch_size, shuffle, seed
        self.drop_last, self.prefetch_tiles, self.shared_memory = drop_last, prefetch_tiles, shared_memory
        self.num_replicas, self.rank = num_replicas, rank
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _replica(self) -> tuple[int, int]:
        if self.num_replicas is not None:
            return self.rank or 0, self.num_replicas
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def my_tiles(self) -> np.ndarray:
        """This (replica, worker)'s tiles for the current epoch, in processing order."""
        n_tiles = self.dataset.n_tiles
        tiles = (
            np.random.default_rng(self.seed + self.epoch).permutation(n_tiles) if self.shuffle else np.arange(n_tiles)
        )
        rank, replicas = self._replica()
        per = -(-n_tiles // replicas)
        tiles = tiles[rank * per : (rank + 1) * per]
        info = torch.utils.data.get_worker_info()
        if info is not None:
            per_worker = -(-len(tiles) // info.num_workers)
            tiles = tiles[info.id * per_worker : (info.id + 1) * per_worker]
        return tiles

    def _batch(self, idx: np.ndarray) -> dict[str, torch.Tensor]:
        ds = self.dataset
        nnz = ds.batch_nnz(idx)
        if self.shared_memory:
            indices = torch.empty(nnz, dtype=torch.int32).share_memory_()
            values = torch.empty(nnz, dtype=torch.float32).share_memory_()
        else:
            indices, values = torch.empty(nnz, dtype=torch.int32), torch.empty(nnz, dtype=torch.float32)
        b = ds.get_batch(idx, out_indices=indices.numpy(), out_values=values.numpy())
        return {"indptr": torch.from_numpy(b.indptr.astype(np.int32)), "indices": indices, "values": values}

    def __iter__(self):
        ds = self.dataset
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, self.epoch, info.id if info else 0])
        tiles = self.my_tiles()
        ds.prefetch(tiles[: self.prefetch_tiles + 1])
        buffer = np.empty(0, dtype=np.int64)
        for k, t in enumerate(tiles):
            ds.prefetch(tiles[k + 1 : k + 1 + self.prefetch_tiles])
            lo, hi = ds.tile_bounds(int(t))
            cells = np.arange(lo, hi, dtype=np.int64)
            if self.shuffle:
                rng.shuffle(cells)
            buffer = np.concatenate([buffer, cells])
            while len(buffer) >= self.batch_size:
                yield self._batch(buffer[: self.batch_size])
                buffer = buffer[self.batch_size :]
        if len(buffer) and not self.drop_last:
            yield self._batch(buffer)


def to_torch_csr(batch: dict[str, torch.Tensor], n_genes: int) -> torch.Tensor:
    """Rebuild the CSR tensor of a batch yielded by :class:`TileShuffleDataset` (typically in the training process)."""
    return torch.sparse_csr_tensor(
        batch["indptr"],
        batch["indices"],
        batch["values"],
        size=(len(batch["indptr"]) - 1, n_genes),
        check_invariants=False,
    )
