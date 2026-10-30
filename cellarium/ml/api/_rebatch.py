# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Reading the cells of a :class:`~cellarium.ml.api.CellariumDataView` one shard at a time, for passes over cells that are
scattered over many shards of the data: a :class:`ShardChunkDataset` of the dataloader workers, whose items
:class:`RebatchedDataset` puts together into batches, and the datamodule that uses them.
"""

import math
import random
import warnings
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import scipy.sparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset

from cellarium.ml.api._view_collections import RamCollection, ViewCollection, unwrap_collection
from cellarium.ml.core.datamodule import CellariumAnnDataDataModule
from cellarium.ml.data import DistributedCollection, IterableDistributedAnnDataCollectionDataset
from cellarium.ml.utilities.distributed import get_rank_and_num_replicas, get_worker_info

Batch = dict[str, Any]


class ShardChunkDataset(IterableDistributedAnnDataCollectionDataset):
    r"""
    The cells of a collection read one shard at a time: every item that this dataset yields is *all* the cells of one
    shard (restricted to ``[start_idx, end_idx)``), instead of ``batch_size`` cells. For a view of cells that are
    scattered over the data, a batch of ``batch_size`` cells would have cells in many shards; here every shard is
    read and decoded once per pass, by the worker that is assigned the shard, and the shards are announced to the
    collection ahead of time as for any other dataset. The items are meant to be put together into batches of exactly
    ``batch_size`` cells by :class:`RebatchedDataset`, around a ``DataLoader`` of this dataset.

    The order of the cells is that of :class:`IterableDistributedAnnDataCollectionDataset` for a single process
    (shards, then the cells within each shard, are shuffled with the same generator), so with one worker the batches
    of the two datasets are the same. With several workers, ``same_order`` gives shard ``i`` to worker
    ``i % num_workers`` (the dataloader returns the items in the order of the shards) and ``cache_efficient`` gives
    each worker a contiguous run of shards. This dataset is for one process (the replicas of a distributed run would
    need other shards), and resuming from a checkpoint is not supported.

    The arguments are those of :class:`IterableDistributedAnnDataCollectionDataset`.
    """

    def _chunks(self) -> list[list[int]]:
        """The cells of each shard in the order of the epoch, shard by shard."""
        limits = [idx for idx in self.dadc.limits if idx > self.start_idx and idx < self.end_idx]
        iter_limits = list(zip([self.start_idx] + limits, limits + [self.end_idx]))
        if not self.shuffle:
            return [list(range(lower, upper)) for lower, upper in iter_limits]
        rng = torch.Generator()
        rng.manual_seed(self.shuffle_seed + self.epoch)
        chunks = []
        for shard in torch.randperm(len(iter_limits), generator=rng).tolist():
            lower, upper = iter_limits[shard]
            chunks.append((torch.randperm(upper - lower, generator=rng) + lower).tolist())
        return chunks

    def __iter__(self):
        _, num_replicas = get_rank_and_num_replicas()
        if num_replicas != 1:
            raise NotImplementedError("ShardChunkDataset reads for a single process, not for a distributed run.")
        if self.test_mode and isinstance(self.dadc, DistributedCollection):
            self.dadc.reset_cache()

        worker_id, num_workers = get_worker_info()
        if self.worker_seed is not None:
            current_worker_seed = self.worker_seed + self.epoch * num_workers + worker_id
            random.seed(current_worker_seed)
            np.random.seed(current_worker_seed)
            torch.manual_seed(current_worker_seed)

        chunks = self._chunks()
        if self.iteration_strategy == "same_order":
            mine = chunks[worker_id::num_workers]
        else:
            per_worker = math.ceil(len(chunks) / num_workers)
            mine = chunks[worker_id * per_worker : (worker_id + 1) * per_worker]

        yield from self._read_batches(mine)

        self.set_epoch(self.epoch + 1)
        self.set_resume_step(None)


def _is_shared(key: str) -> bool:
    """Whether the value of ``key`` is the same for every cell (as in :func:`~cellarium.ml.utilities.data.collate_fn`)
    and so is not concatenated."""
    return key.endswith("_g") or key.endswith("_categories")


def _make_csr(crow: torch.Tensor, col: torch.Tensor, values: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    with warnings.catch_warnings(), torch.sparse.check_sparse_tensor_invariants(False):
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
        return torch.sparse_csr_tensor(crow, col, values, size=size, dtype=values.dtype, device=values.device)


def _n_rows(value: Any) -> int:
    return value.shape[0] if hasattr(value, "shape") else len(value)


def _slice_rows(value: Any, start: int, stop: int) -> Any:
    """The rows ``start:stop`` of a value that has one row per cell."""
    if isinstance(value, torch.Tensor) and value.layout == torch.sparse_csr:
        crow = value.crow_indices()
        first, last = int(crow[start]), int(crow[stop])
        return _make_csr(
            crow[start : stop + 1] - first,
            value.col_indices()[first:last],
            value.values()[first:last],
            (stop - start, value.shape[1]),
        )
    if isinstance(value, torch.Tensor) and value.layout == torch.sparse_coo:
        value = value.coalesce()
        indices, values = value.indices(), value.values()
        keep = (indices[0] >= start) & (indices[0] < stop)
        indices = indices[:, keep].clone()
        indices[0] -= start
        return torch.sparse_coo_tensor(indices, values[keep], size=(stop - start, *value.shape[1:]))
    return value[start:stop]


def _concatenate_rows(values: list[Any]) -> Any:
    """The rows of all the ``values`` (each with one row per cell), one after the other."""
    if len(values) == 1:
        return values[0]
    first = values[0]
    if isinstance(first, torch.Tensor) and first.layout == torch.sparse_csr:
        crows, offset = [first.crow_indices()], int(first.crow_indices()[-1])
        for value in values[1:]:
            crow = value.crow_indices()
            crows.append(crow[1:] + offset)
            offset += int(crow[-1])
        return _make_csr(
            torch.cat(crows),
            torch.cat([v.col_indices() for v in values]),
            torch.cat([v.values() for v in values]),
            (sum(v.shape[0] for v in values), first.shape[1]),
        )
    if isinstance(first, torch.Tensor):
        return torch.cat(values, dim=0)
    if scipy.sparse.issparse(first):
        return scipy.sparse.vstack(values, format=first.format)
    if isinstance(first, list):
        return [item for value in values for item in value]
    return np.concatenate(values, axis=0)


def _per_cell_leaves(batch: Batch) -> Iterator[Any]:
    for key, value in batch.items():
        if _is_shared(key):
            continue
        if isinstance(value, dict):
            yield from value.values()
        else:
            yield value


def _n_cells(batch: Batch) -> int:
    """The number of cells of a batch of fields (those with one row per cell)."""
    for value in _per_cell_leaves(batch):
        return _n_rows(value)
    raise ValueError("The batch has no field with a row for each cell, so its cells cannot be counted.")


def _map_cells(batch: Batch, fn: Callable[[str, list[Any] | Any], Any], batches: list[Batch]) -> Batch:
    """A batch with the value of each field made by ``fn(key, [that field of each of the batches])``; fields that
    are the same for every cell are taken from the first batch."""
    out: Batch = {}
    for key, value in batch.items():
        if _is_shared(key):
            out[key] = value
        elif isinstance(value, dict):
            out[key] = {subkey: fn(subkey, [b[key][subkey] for b in batches]) for subkey in value}
        else:
            out[key] = fn(key, [b[key] for b in batches])
    return out


def concatenate_batches(batches: list[Batch]) -> Batch:
    """One batch of the cells of all the ``batches`` (dictionaries of fields as the dataloader makes them), in order."""
    if len(batches) == 1:
        return batches[0]
    return _map_cells(batches[0], lambda _, values: _concatenate_rows(values), batches)


def slice_batch(batch: Batch, start: int, stop: int) -> Batch:
    """The cells ``start:stop`` of a batch."""
    return _map_cells(batch, lambda _, values: _slice_rows(values[0], start, stop), [batch])


def _passthrough(batch: Batch) -> Batch:
    return batch


class RebatchedDataset(IterableDataset):
    """
    Batches of exactly ``batch_size`` cells (the last one has fewer, unless ``drop_incomplete_batch``), made by putting
    together the items of a ``DataLoader`` of a :class:`ShardChunkDataset`, which come in any
    sizes. The items are whatever the loader's ``collate_fn`` makes of them (so the transforms of the workers have been
    applied), and this only concatenates them along the cells and cuts the batches: for each field, the rows (of
    tensors, sparse CSR and COO tensors, arrays, scipy sparse matrices and lists) are the cells, except for the fields
    that are the same for every cell (the names ending in ``_g`` or ``_categories``), which are taken from the first
    item. The cells keep the order in which the loader returns them.

    Wrap it in a ``DataLoader`` with ``batch_size=None`` and no workers, which :func:`make_loader` does.

    Args:
        loader: The ``DataLoader`` of a :class:`ShardChunkDataset`.
        batch_size: The number of cells per batch.
        drop_incomplete_batch: Whether to leave out the last batch if it has fewer than ``batch_size`` cells.
    """

    def __init__(self, loader: DataLoader, batch_size: int, drop_incomplete_batch: bool = False) -> None:
        if not isinstance(loader.dataset, ShardChunkDataset):
            raise TypeError(f"The loader must read a ShardChunkDataset, not {type(loader.dataset).__name__}.")
        self.loader = loader
        self.batch_size = batch_size
        self.drop_incomplete_batch = drop_incomplete_batch

    def __len__(self) -> int:
        n_obs = self.loader.dataset.end_idx - self.loader.dataset.start_idx  # type: ignore[attr-defined]
        if self.drop_incomplete_batch:
            return n_obs // self.batch_size
        return -(-n_obs // self.batch_size)

    def set_epoch(self, epoch: int) -> None:
        self.loader.dataset.set_epoch(epoch)  # type: ignore[attr-defined]

    def __iter__(self) -> Iterator[Batch]:
        pending: list[tuple[Batch, int]] = []  # the items not yet in a batch, with their numbers of cells
        n_pending = 0
        for chunk in self.loader:
            size = _n_cells(chunk)
            pending.append((chunk, size))
            n_pending += size
            while n_pending >= self.batch_size:
                taken: list[Batch] = []
                needed = self.batch_size
                while needed > 0:
                    item, size = pending[0]
                    if size <= needed:
                        taken.append(item)
                        pending.pop(0)
                        needed -= size
                    else:
                        taken.append(slice_batch(item, 0, needed))
                        pending[0] = (slice_batch(item, needed, size), size - needed)
                        needed = 0
                n_pending -= self.batch_size
                yield concatenate_batches(taken)
        if pending and not self.drop_incomplete_batch:
            yield concatenate_batches([item for item, _ in pending])


def make_loader(
    dataset: ShardChunkDataset,
    *,
    collate_fn: Callable,
    num_workers: int = 0,
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
    pin_memory: bool = False,
) -> DataLoader:
    """
    A ``DataLoader`` of batches of exactly ``dataset.batch_size`` cells of a :class:`ShardChunkDataset`. The workers
    (``num_workers`` of them) read the dataset a shard at a time and make each shard's batch with ``collate_fn``, as
    they do for any dataset; the batches are made from these in the main process by :class:`RebatchedDataset`, which
    does nothing else. The loader has the length (the number of batches) and the ``set_epoch`` that Lightning needs.
    """
    inner = DataLoader(
        dataset,
        num_workers=num_workers,
        collate_fn=collate_fn,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )
    rebatched = RebatchedDataset(inner, dataset.batch_size, dataset.drop_incomplete_batch)
    # batch_size=None: the collate_fn is given one batch (not a list), and must leave it as it is
    return DataLoader(
        rebatched,
        batch_size=None,
        num_workers=0,
        collate_fn=_passthrough,  # type: ignore[arg-type]
        pin_memory=pin_memory,
    )


class ShardChunkDataModule(CellariumAnnDataDataModule):
    """
    A datamodule that reads the cells of a :class:`ViewCollection` of cells that are not in memory a shard at a time
    (with :class:`ShardChunkDataset`, whose items :class:`RebatchedDataset` puts together into batches), in a single
    process. It reads other collections (cells in memory have no shards to read), and in a distributed run, as a
    :class:`~cellarium.ml.core.CellariumAnnDataDataModule` does.
    """

    def reads_by_shard(self) -> bool:
        """Whether the cells are read one shard at a time."""
        if not isinstance(self.dadc, ViewCollection) or isinstance(unwrap_collection(self.dadc), RamCollection):
            return False
        return not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1)

    def setup(self, stage: str | None = None) -> None:
        super().setup(stage)
        if not self.reads_by_shard():
            return
        names = {"fit": ["train", "val"], "validate": ["val"], "predict": ["predict"], "test": ["test"]}
        for name in names.get(stage or "", []):
            dataset = getattr(self, f"{name}_dataset")
            setattr(
                self,
                f"{name}_dataset",
                ShardChunkDataset(
                    dadc=dataset.dadc,
                    batch_keys=dataset.batch_keys,
                    batch_size=dataset.batch_size,
                    iteration_strategy=dataset.iteration_strategy,
                    shuffle=dataset.shuffle,
                    shuffle_seed=dataset.shuffle_seed,
                    drop_last_indices=dataset.drop_last_indices,
                    drop_incomplete_batch=dataset.drop_incomplete_batch,
                    start_idx=dataset.start_idx,
                    end_idx=dataset.end_idx,
                    worker_seed=dataset.worker_seed,
                    test_mode=dataset.test_mode,
                    prefetch_lookahead=dataset.prefetch_lookahead,
                ),
            )

    def _chunk_loader(self, dataset: ShardChunkDataset) -> DataLoader:
        self._check_num_workers()
        return make_loader(
            dataset,
            collate_fn=self._effective_collate_fn(),
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self) -> DataLoader:
        if isinstance(self.train_dataset, ShardChunkDataset):
            return self._chunk_loader(self.train_dataset)
        return super().train_dataloader()

    def val_dataloader(self) -> DataLoader:
        if isinstance(self.val_dataset, ShardChunkDataset):
            return self._chunk_loader(self.val_dataset)
        return super().val_dataloader()

    def predict_dataloader(self) -> DataLoader:
        if isinstance(self.predict_dataset, ShardChunkDataset):
            return self._chunk_loader(self.predict_dataset)
        return super().predict_dataloader()

    def test_dataloader(self) -> DataLoader:
        if isinstance(self.test_dataset, ShardChunkDataset):
            return self._chunk_loader(self.test_dataset)
        return super().test_dataloader()
