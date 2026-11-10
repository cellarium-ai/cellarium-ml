# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from abc import abstractmethod
from typing import Any

import lightning.pytorch as pl
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from cellarium.ml.models.model import CellariumModel
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)

# ----------------------------------------------------------------------
# Tensor storage of the Plaid sketch
# ----------------------------------------------------------------------
#
# The Plaid sketch keeps one row per occupied grid voxel, as a struct of arrays living on a single device (so
# that inserting a minibatch and coarsening the grid are a handful of sort / scatter / gather kernels rather
# than a Python loop over voxels):
#
# * ``coords`` - integer grid coordinates of the voxel, ``(V, D)``.
# * ``hashes`` - a 64-bit hash of ``coords`` (a random linear form, wrapping modulo ``2**64``), ``(V,)``. Voxels
#   are identified by their hash alone. With 2 million voxels the chance of any two colliding is about ``1e-7``.
# * ``seen`` - the number of cells ever assigned to the voxel, ``(V,)``.
# * ``slot_tag`` / ``slot_id`` - the up to ``K`` retained cells of the voxel, ``(V, K)``, sorted by tag.
#
# Reservoir sampling by random tags: every cell is given an independent Uniform(0, 1) tag when it is first
# seen, and a voxel retains the K cells with the smallest tags among all cells that ever landed in it. That is
# exactly a uniform sample of K of those cells (without replacement), and it is *mergeable*: the K smallest tags
# of a union of two voxels are among the two voxels' own K smallest. So both inserting a minibatch and merging
# voxels when the grid is coarsened are "union the candidates, keep the K smallest per voxel", which is two
# stable sorts, with no sequential per-cell logic.
#
# Cells are referred to by integer ids; the (host-side) payload of the cells that currently hold a slot (their
# names and, optionally, their data) lives in a _CellPool.

_NO_TAG = float("inf")
_NO_CELL = -1


def _bottom_k(
    group: torch.Tensor, tag: torch.Tensor, cell_id: torch.Tensor, n_groups: int, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For every group keep the ``k`` entries with the smallest tags.

    Args:
        group: Group of every entry, ``(M,)`` integers in ``[0, n_groups)``.
        tag: Random tag of every entry, ``(M,)`` floats.
        cell_id: Cell id of every entry, ``(M,)`` integers.
        n_groups: Number of groups.
        k: Number of entries kept per group.

    Returns:
        ``slot_tag`` and ``slot_id``, both ``(n_groups, k)``: each row holds the kept entries in ascending tag
        order, padded with ``inf`` / ``-1``.
    """
    device = tag.device
    slot_tag = torch.full((n_groups, k), _NO_TAG, dtype=tag.dtype, device=device)
    slot_id = torch.full((n_groups, k), _NO_CELL, dtype=torch.int64, device=device)
    if tag.numel() == 0:
        return slot_tag, slot_id

    order = torch.argsort(tag, stable=True)
    group, tag, cell_id = group[order], tag[order], cell_id[order]
    order = torch.argsort(group, stable=True)
    group, tag, cell_id = group[order], tag[order], cell_id[order]

    counts = torch.bincount(group, minlength=n_groups)
    starts = torch.cumsum(counts, dim=0) - counts
    rank = torch.arange(group.numel(), device=device) - starts[group]
    keep = rank < k
    slot_tag[group[keep], rank[keep]] = tag[keep]
    slot_id[group[keep], rank[keep]] = cell_id[keep]
    return slot_tag, slot_id


def _shifted_hashes(coords: torch.Tensor, shifts: torch.Tensor | None, multipliers: torch.Tensor) -> torch.Tensor:
    """
    Hash the rows of ``coords >> shifts``, in row chunks so that no ``(V, D)`` temporary is created.

    An arithmetic right shift is a floor division by a power of two, i.e. exactly the coordinate of the
    containing voxel after doubling the voxel size along an axis ``shift`` times.
    """
    n, d = coords.shape
    out = torch.empty(n, dtype=torch.int64, device=coords.device)
    chunk = max(1, (1 << 24) // max(d, 1))
    for start in range(0, n, chunk):
        c = coords[start : start + chunk]
        if shifts is not None:
            c = c >> shifts
        out[start : start + chunk] = (c * multipliers).sum(dim=1)
    return out


def _first_true(pred: Any, hi: int) -> int:
    """
    Smallest ``s`` in ``[1, hi]`` with ``pred(s)`` true, or ``hi + 1`` if there is none.

    ``pred`` must be monotone (false up to some point, true afterwards) and is assumed false at ``0``. The
    search gallops (probes 1, 3, 7, ...) before bisecting, because the answer is usually small.
    """
    lo, step = 0, 1
    while True:
        probe = min(lo + step, hi)
        if pred(probe):
            break
        if probe >= hi:
            return hi + 1
        lo, step = probe, step * 2
    high = probe
    while high - lo > 1:
        mid = (lo + high) // 2
        if pred(mid):
            high = mid
        else:
            lo = mid
    return high


class _VoxelStore:
    """
    Occupied voxels of an integer grid, each with a bottom-``K`` (by random tag) reservoir of cell ids.

    Args:
        dim: Number of grid dimensions ``D``.
        slots: Number of cells ``K`` retained per voxel.
        seed: Seed of the random hash multipliers.
        device: Device holding all the state.
    """

    # The sorted lookup index is a large "main" run plus a small "delta" run holding the voxels added since
    # the main run was last rebuilt, so that adding a minibatch of voxels does not re-sort every voxel.
    _MIN_DELTA = 16384

    def __init__(self, dim: int, slots: int, seed: int, device: torch.device | str) -> None:
        self.dim = dim
        self.slots = slots
        self.seed = seed
        generator = torch.Generator().manual_seed(seed)
        multipliers = torch.randint(-(1 << 62), 1 << 62, (dim,), generator=generator, dtype=torch.int64) | 1
        self.multipliers = multipliers.to(device)
        self.n = 0
        self._allocate(0)
        self._reset_index()
        self._meta_chunks: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._meta_pending = 0
        self._meta_size = 0
        self.has_metadata = False

    # Storage management

    @property
    def device(self) -> torch.device:
        return self.coords.device

    @property
    def capacity(self) -> int:
        return self.coords.shape[0]

    def _allocate(self, capacity: int) -> None:
        """Replace the arrays by fresh ones with room for ``capacity`` voxels, keeping the first ``n`` rows."""
        device = self.multipliers.device
        old = None if self.n == 0 else self._rows(self.n)
        self.coords = torch.zeros((capacity, self.dim), dtype=torch.int64, device=device)
        self.hashes = torch.zeros(capacity, dtype=torch.int64, device=device)
        self.seen = torch.zeros(capacity, dtype=torch.int64, device=device)
        self.slot_tag = torch.full((capacity, self.slots), _NO_TAG, dtype=torch.float64, device=device)
        self.slot_id = torch.full((capacity, self.slots), _NO_CELL, dtype=torch.int64, device=device)
        if old is not None:
            self._write(slice(0, self.n), *old)

    def _rows(self, n: int) -> tuple[torch.Tensor, ...]:
        return (self.coords[:n], self.hashes[:n], self.seen[:n], self.slot_tag[:n], self.slot_id[:n])

    def _write(
        self,
        where: slice,
        coords: torch.Tensor,
        hashes: torch.Tensor,
        seen: torch.Tensor,
        slot_tag: torch.Tensor,
        slot_id: torch.Tensor,
    ) -> None:
        self.coords[where] = coords
        self.hashes[where] = hashes
        self.seen[where] = seen
        self.slot_tag[where] = slot_tag
        self.slot_id[where] = slot_id

    def _reserve(self, needed: int, limit: int) -> None:
        if needed <= self.capacity:
            return
        self._allocate(max(needed, min(int(self.capacity * 1.5) + 1024, limit)))

    def to(self, device: torch.device | str) -> None:
        device = torch.device(device)
        if device == self.device:
            return
        self.multipliers = self.multipliers.to(device)
        for name in (
            "coords",
            "hashes",
            "seen",
            "slot_tag",
            "slot_id",
            "main_hash",
            "main_row",
            "delta_hash",
            "delta_row",
        ):
            setattr(self, name, getattr(self, name).to(device))
        self._meta_chunks = [(r.to(device), c.to(device)) for r, c in self._meta_chunks]

    # Lookup index

    def _reset_index(self) -> None:
        empty = torch.empty(0, dtype=torch.int64, device=self.multipliers.device)
        self.main_hash, self.main_row, self.delta_hash, self.delta_row = empty, empty, empty, empty

    def _rebuild_index(self) -> None:
        order = torch.argsort(self.hashes[: self.n])
        self.main_hash = self.hashes[: self.n][order]
        self.main_row = order
        empty = torch.empty(0, dtype=torch.int64, device=self.device)
        self.delta_hash, self.delta_row = empty, empty

    def _lookup(self, hashes: torch.Tensor) -> torch.Tensor:
        """Row of every (sorted, unique) hash, ``-1`` where the voxel does not exist yet."""
        rows = torch.full_like(hashes, -1)
        for run_hash, run_row in ((self.main_hash, self.main_row), (self.delta_hash, self.delta_row)):
            if run_hash.numel() == 0:
                continue
            pos = torch.searchsorted(run_hash, hashes).clamp_(max=run_hash.numel() - 1)
            rows = torch.where(run_hash[pos] == hashes, run_row[pos], rows)
        return rows

    def _add_to_index(self, hashes: torch.Tensor, rows: torch.Tensor) -> None:
        """Add new voxels (``hashes`` ascending) to the index."""
        if self.delta_hash.numel():
            hashes = torch.cat([self.delta_hash, hashes])
            rows = torch.cat([self.delta_row, rows])
            order = torch.argsort(hashes)
            hashes, rows = hashes[order], rows[order]
        self.delta_hash, self.delta_row = hashes, rows
        if hashes.numel() > max(self._MIN_DELTA, self.n // 32):
            hashes = torch.cat([self.main_hash, self.delta_hash])
            rows = torch.cat([self.main_row, self.delta_row])
            order = torch.argsort(hashes)
            self.main_hash, self.main_row = hashes[order], rows[order]
            empty = torch.empty(0, dtype=torch.int64, device=self.device)
            self.delta_hash, self.delta_row = empty, empty

    # Insertion

    @torch.no_grad()
    def insert(
        self,
        coords: torch.Tensor,
        tags: torch.Tensor,
        first_id: int,
        metadata_codes: torch.Tensor | None,
        capacity_limit: int,
    ) -> torch.Tensor:
        """
        Add a minibatch of cells; cell ``i`` gets the id ``first_id + i``.

        Args:
            coords: Voxel coordinates of the cells, ``(B, D)`` int64.
            tags: Random ``Uniform(0, 1)`` tag of every cell, ``(B,)`` float64.
            first_id: Id of the first cell of the batch.
            metadata_codes: Optional integer code of every cell's metadata category, ``(B,)``.
            capacity_limit: Row count the arrays are grown towards (growth never exceeds it unless needed).

        Returns:
            Indices (into the batch) of the cells that hold a slot after this update, ``(W,)``.
        """
        batch = coords.shape[0]
        hashes = _shifted_hashes(coords, None, self.multipliers)
        unique_hashes, inverse = torch.unique(hashes, return_inverse=True)
        n_unique = unique_hashes.numel()
        cell_index = torch.arange(batch, device=self.device)
        representative = torch.full((n_unique,), batch, dtype=torch.int64, device=self.device)
        representative.scatter_reduce_(0, inverse, cell_index, "amin", include_self=True)

        rows = self._lookup(unique_hashes)
        is_new = rows < 0
        n_new = int(is_new.sum())
        if n_new:
            start = self.n
            self._reserve(start + n_new, capacity_limit)
            new_rows = torch.arange(start, start + n_new, device=self.device)
            rows[is_new] = new_rows
            self.coords[start : start + n_new] = coords[representative[is_new]]
            self.hashes[start : start + n_new] = unique_hashes[is_new]
            self.seen[start : start + n_new] = 0
            self.slot_tag[start : start + n_new] = _NO_TAG
            self.slot_id[start : start + n_new] = _NO_CELL
            self.n += n_new
            self._add_to_index(unique_hashes[is_new], new_rows)

        self.seen[rows] += torch.bincount(inverse, minlength=n_unique)

        # Candidates of every touched voxel: its current slots and the batch's cells.
        k = self.slots
        old_id = self.slot_id[rows].reshape(-1)
        old_tag = self.slot_tag[rows].reshape(-1)
        old_group = torch.arange(n_unique, device=self.device).repeat_interleave(k)
        has_old = old_id >= 0
        group = torch.cat([old_group[has_old], inverse])
        tag = torch.cat([old_tag[has_old], tags])
        cell_id = torch.cat([old_id[has_old], first_id + cell_index])
        new_tag, new_id = _bottom_k(group, tag, cell_id, n_unique, k)
        self.slot_tag[rows] = new_tag
        self.slot_id[rows] = new_id

        if metadata_codes is not None:
            pairs = torch.unique(torch.stack([rows[inverse], metadata_codes]), dim=1)
            self._meta_chunks.append((pairs[0], pairs[1]))
            self._meta_pending += pairs.shape[1]
            self.has_metadata = True
            if self._meta_pending > max(65536, 2 * self._meta_size):
                self._dedupe_metadata()

        return new_id[new_id >= first_id] - first_id

    # Metadata categories

    def _dedupe_metadata(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Collapse the pending ``(row, category code)`` pairs into one duplicate-free pair of arrays."""
        if not self._meta_chunks:
            empty = torch.empty(0, dtype=torch.int64, device=self.device)
            return empty, empty
        rows = torch.cat([r for r, _ in self._meta_chunks])
        codes = torch.cat([c for _, c in self._meta_chunks])
        n_codes = int(codes.max()) + 1 if codes.numel() else 1
        keys = torch.unique(rows * n_codes + codes)
        rows, codes = keys // n_codes, keys % n_codes
        self._meta_chunks = [(rows, codes)]
        self._meta_pending = 0
        self._meta_size = rows.numel()
        return rows, codes

    def diversity(self) -> torch.Tensor:
        """Number of distinct metadata categories seen by every voxel, ``(n,)``."""
        rows, _ = self._dedupe_metadata()
        return torch.bincount(rows, minlength=self.n)

    # Coarsening

    def _max_doublings(self) -> int:
        """Number of right shifts after which every coordinate of every axis is ``0`` or ``-1``."""
        if self.n == 0:
            return 1
        largest = int(self.coords[: self.n].abs().max())
        return min(max(largest, 1).bit_length() + 1, 62)

    def coarsening_order(self, voxel_size: torch.Tensor) -> torch.Tensor:
        """
        Axes in the order in which they get doubled: always the axis with the smallest current voxel size,
        ties to the lowest axis index. Entry ``j`` is the axis doubled in step ``j + 1``.
        """
        m = self._max_doublings()
        sizes = voxel_size.to(device=self.device, dtype=torch.float64)
        powers = 2.0 ** torch.arange(m, device=self.device, dtype=torch.float64)
        events = (sizes[:, None] * powers[None, :]).reshape(-1)
        return torch.argsort(events, stable=True) // m

    def shifts_after(self, order: torch.Tensor, steps: int) -> torch.Tensor:
        """How many times every axis has been doubled after the first ``steps`` steps of ``order``."""
        return torch.bincount(order[:steps], minlength=self.dim)

    def count_after(self, shifts: torch.Tensor) -> int:
        """Number of occupied voxels after doubling the voxel size ``shifts[d]`` times along every axis ``d``."""
        return torch.unique(_shifted_hashes(self.coords[: self.n], shifts, self.multipliers)).numel()

    @torch.no_grad()
    def _coarsened_arrays(
        self, shifts: torch.Tensor, capacity: int
    ) -> tuple[int, tuple[torch.Tensor, ...], torch.Tensor]:
        """Merged rows (and the old-row -> new-row map) after doubling ``shifts[d]`` times along every axis."""
        n, k = self.n, self.slots
        hashes = _shifted_hashes(self.coords[:n], shifts, self.multipliers)
        unique_hashes, inverse = torch.unique(hashes, return_inverse=True)
        n_new = unique_hashes.numel()
        representative = torch.full((n_new,), n, dtype=torch.int64, device=self.device)
        representative.scatter_reduce_(0, inverse, torch.arange(n, device=self.device), "amin", include_self=True)

        capacity = max(capacity, n_new)
        coords = torch.zeros((capacity, self.dim), dtype=torch.int64, device=self.device)
        coords[:n_new] = self.coords[representative] >> shifts
        new_hashes = torch.zeros(capacity, dtype=torch.int64, device=self.device)
        new_hashes[:n_new] = unique_hashes
        seen = torch.zeros(capacity, dtype=torch.int64, device=self.device)
        seen[:n_new].index_add_(0, inverse, self.seen[:n])

        old_id = self.slot_id[:n].reshape(-1)
        has_old = old_id >= 0
        group = inverse.repeat_interleave(k)
        merged_tag, merged_id = _bottom_k(
            group[has_old], self.slot_tag[:n].reshape(-1)[has_old], old_id[has_old], n_new, k
        )
        slot_tag = torch.full((capacity, k), _NO_TAG, dtype=torch.float64, device=self.device)
        slot_id = torch.full((capacity, k), _NO_CELL, dtype=torch.int64, device=self.device)
        slot_tag[:n_new] = merged_tag
        slot_id[:n_new] = merged_id
        return n_new, (coords, new_hashes, seen, slot_tag, slot_id), inverse

    @torch.no_grad()
    def merge_to(self, shifts: torch.Tensor, capacity: int) -> None:
        """Coarsen in place: double the voxel size ``shifts[d]`` times along every axis, merging voxels."""
        n_new, arrays, inverse = self._coarsened_arrays(shifts, capacity)
        self.coords, self.hashes, self.seen, self.slot_tag, self.slot_id = arrays
        self.n = n_new
        if self._meta_chunks:
            rows, codes = self._dedupe_metadata()
            self._meta_chunks = [(inverse[rows], codes)]
            self._dedupe_metadata()
        self._rebuild_index()

    @torch.no_grad()
    def coarsened(self, shifts: torch.Tensor) -> "_VoxelStore":
        """A coarsened copy (without metadata); this store is left untouched."""
        n_new, arrays, _ = self._coarsened_arrays(shifts, 0)
        other = _VoxelStore.__new__(_VoxelStore)
        other.dim, other.slots, other.seed = self.dim, self.slots, self.seed
        other.multipliers = self.multipliers
        other.n = n_new
        other.coords, other.hashes, other.seen, other.slot_tag, other.slot_id = arrays
        other._reset_index()
        other._meta_chunks, other._meta_pending, other._meta_size, other.has_metadata = [], 0, 0, False
        return other

    @torch.no_grad()
    def filter(self, keep: torch.Tensor) -> None:
        """Drop the voxels (rows) for which ``keep`` is false."""
        kept = keep.nonzero().squeeze(1)
        n_kept = kept.numel()
        if self._meta_chunks:
            rows, codes = self._dedupe_metadata()
            new_row = torch.full((self.n,), -1, dtype=torch.int64, device=self.device)
            new_row[kept] = torch.arange(n_kept, device=self.device)
            rows = new_row[rows]
            alive = rows >= 0
            self._meta_chunks = [(rows[alive], codes[alive])]
            self._meta_size = int(alive.sum())
        self._write(slice(0, n_kept), *(a[kept] for a in self._rows(self.n)))
        self.n = n_kept
        self._rebuild_index()

    # Summaries and serialization

    def total_cells(self) -> int:
        """Number of cells holding a slot (every voxel holds ``min(seen, K)`` of them)."""
        return int(torch.clamp(self.seen[: self.n], max=self.slots).sum())

    def state_dict(self) -> dict[str, Any]:
        rows, codes = self._dedupe_metadata()
        n = self.n
        return {
            "dim": self.dim,
            "slots": self.slots,
            "seed": self.seed,
            "coords": self.coords[:n].cpu(),
            "seen": self.seen[:n].cpu(),
            "slot_tag": self.slot_tag[:n].cpu(),
            "slot_id": self.slot_id[:n].cpu(),
            "meta_rows": rows.cpu(),
            "meta_codes": codes.cpu(),
            "has_metadata": self.has_metadata,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any], device: torch.device | str) -> "_VoxelStore":
        store = cls(state["dim"], state["slots"], state["seed"], device)
        n = state["coords"].shape[0]
        store._allocate(n)
        store.n = n
        store.coords[:] = state["coords"].to(device)
        store.seen[:] = state["seen"].to(device)
        store.slot_tag[:] = state["slot_tag"].to(device)
        store.slot_id[:] = state["slot_id"].to(device)
        store.hashes[:] = _shifted_hashes(store.coords, None, store.multipliers)
        store._rebuild_index()
        store.has_metadata = state["has_metadata"]
        if state["meta_rows"].numel():
            store._meta_chunks = [(state["meta_rows"].to(device), state["meta_codes"].to(device))]
            store._meta_size = state["meta_rows"].numel()
        return store


class _CellPool:
    """
    Host-side payload of the cells that hold (or recently held) a slot: names and, optionally, a row of data.

    Cells are appended in increasing id order, so the pool is always sorted by id and cells are found by binary
    search. Cells that were displaced from their voxel's slots stay until :meth:`compact` drops them.
    """

    def __init__(self) -> None:
        self._ids: list[np.ndarray] = []
        self._names: list[np.ndarray] = []
        self._rows: list[sp.csr_matrix | None] = []
        self._len = 0

    def __len__(self) -> int:
        return self._len

    def append(self, ids: np.ndarray, names: np.ndarray, rows: sp.csr_matrix | None) -> None:
        if len(ids) == 0:
            return
        self._ids.append(ids)
        self._names.append(names)
        self._rows.append(rows)
        self._len += len(ids)

    def _consolidate(self) -> None:
        if len(self._ids) <= 1:
            return
        rows = None if self._rows[0] is None else sp.vstack(self._rows, format="csr")
        self._ids = [np.concatenate(self._ids)]
        self._names = [np.concatenate(self._names)]
        self._rows = [rows]

    def take(self, ids: np.ndarray) -> tuple[np.ndarray, sp.csr_matrix | None]:
        """Names and rows of the cells with the given ids, in the given order."""
        if len(ids) == 0 or self._len == 0:
            return np.empty(0, dtype=object), None
        self._consolidate()
        pool_ids = self._ids[0]
        pos = np.searchsorted(pool_ids, ids)
        if (pool_ids[np.minimum(pos, len(pool_ids) - 1)] != ids).any():
            raise KeyError("a retained cell is missing from the cell pool")
        rows = self._rows[0]
        return self._names[0][pos], None if rows is None else rows[pos]

    def compact(self, live_ids: np.ndarray) -> None:
        """Keep only the cells with the given (sorted) ids."""
        if self._len == 0:
            return
        self._consolidate()
        pool_ids = self._ids[0]
        keep = np.zeros(len(pool_ids), dtype=bool)
        keep[np.searchsorted(pool_ids, live_ids)] = True
        rows = self._rows[0]
        self._ids = [pool_ids[keep]]
        self._names = [self._names[0][keep]]
        self._rows = [None if rows is None else rows[keep]]
        self._len = int(keep.sum())

    def state_dict(self) -> dict[str, Any]:
        self._consolidate()
        if not self._ids:
            return {"ids": np.empty(0, dtype=np.int64), "names": [], "csr": None}
        rows = self._rows[0]
        csr = None if rows is None else (rows.indptr, rows.indices, rows.data, rows.shape)
        return {"ids": self._ids[0], "names": [str(n) for n in self._names[0]], "csr": csr}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "_CellPool":
        pool = cls()
        if len(state["ids"]):
            csr = state["csr"]
            rows = None if csr is None else sp.csr_matrix((csr[2], csr[1], csr[0]), shape=csr[3])
            names = np.empty(len(state["names"]), dtype=object)
            names[:] = state["names"]
            pool.append(state["ids"], names, rows)
        return pool


def _to_scipy_csr(x_nk: torch.Tensor) -> sp.csr_matrix:
    csr = x_nk.cpu().to_sparse_csr()
    return sp.csr_matrix(
        (csr.values().numpy(), csr.col_indices().numpy(), csr.crow_indices().numpy()), shape=tuple(csr.shape)
    )


class StreamingGeometricSketch(CellariumModel):
    """
    Abstract base class for streaming geometric sketch models.

    Provides the shared constructor arguments, common Lightning hooks, and
    :meth:`reset_parameters`.  Concrete subclasses own the storage of the
    sketch, and must implement :meth:`update` (the core hashing / binning
    algorithm), :meth:`forward`, :meth:`get_reservoir`,
    :meth:`apply_bucket_filters`, the :attr:`total_cells` and
    :attr:`num_filled_buckets` properties, and the storage hooks
    :meth:`_reset_storage`, :meth:`_storage_state` and :meth:`_load_storage_state`.

    Args:
        var_names_g:
            Gene names for input validation.
        max_cells_per_bucket:
            Maximum cells retained per bucket via uniform reservoir sampling.
        min_cells_per_bucket:
            Density threshold for end-of-training pruning via
            :meth:`apply_bucket_filters`. Buckets that observed fewer cells
            in total are dropped. Default of ``1`` disables filtering.
        min_metadata_diversity:
            Diversity threshold for end-of-training pruning via
            :meth:`apply_bucket_filters`. Buckets that observed fewer unique
            metadata categories are dropped. Default of ``1`` disables filtering.
        store_cell_data:
            If ``True``, accumulate sparse cell expression vectors.
            If ``False``, only cell IDs (``obs_names``) are stored; calling
            ``get_reservoir(return_cell_data=True)`` will raise.
        projector:
            Optional frozen encoder mapping ``(N, G) → (N, D)``. When given,
            its output feeds the bucketing algorithm rather than raw gene expression.
            The module's gradients are disabled on assignment.
        limit_input_to_top_pcs:
            If specified, limit the input to this number of top principal components.
        seed:
            Random seed for reservoir sampling. Sampling draws from a
            generator owned by this model, so results are unaffected by
            other consumers of the global :mod:`torch` RNG.
    """

    def __init__(
        self,
        var_names_g: np.ndarray,
        max_cells_per_bucket: int,
        min_cells_per_bucket: int,
        min_metadata_diversity: int,
        store_cell_data: bool,
        projector: nn.Module | None,
        limit_input_to_top_pcs: int | None,
        seed: int,
    ) -> None:
        super().__init__()

        self.var_names_g = var_names_g
        self.max_cells_per_bucket = max_cells_per_bucket
        self.min_cells_per_bucket = min_cells_per_bucket
        self.min_metadata_diversity = min_metadata_diversity
        self.store_cell_data = store_cell_data
        self.limit_input_to_top_pcs = limit_input_to_top_pcs
        self._seed = seed

        if projector is not None:
            self.projector: nn.Module | None = projector
            self.projector.requires_grad_(False)
        else:
            self.projector = None

        # DDP requires at least one parameter with requires_grad=True even when no
        # optimizer is used; this scalar satisfies that constraint without affecting results.
        self._dummy_param = nn.Parameter(torch.empty(()))

        # Reservoir sampling draws from this generator rather than the global torch RNG,
        # so the sketch is reproducible regardless of other RNG consumers in the process.
        # It is seeded (and re-seeded) in reset_parameters().
        self._generator = torch.Generator()

        # _batches_seen is tracked here because global_step does not increment for
        # models that do not call optimizer.step() (automatic_optimization=False).
        # The storage and EMA state are initialized by reset_parameters().
        self._prev_total_cells: int = 0
        self._ema_delta: float = 0.0
        self.reset_parameters()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def update(
        self,
        x_ng: torch.Tensor,
        obs_names_n: np.ndarray,
        metadata_n: np.ndarray | None = None,
    ) -> int:
        """
        Update the reservoir with a minibatch of cells.

        Args:
            x_ng:
                Expression matrix of shape ``(N, G)``, dense or sparse.
            obs_names_n:
                Cell IDs of shape ``(N,)``.
            metadata_n:
                Optional integer metadata of shape ``(N,)`` used for
                diversity tracking in :meth:`apply_bucket_filters`.

        Returns:
            Number of cells inserted or replaced in this update.
        """

    @abstractmethod
    def get_reservoir(
        self,
        return_cell_data: bool = False,
        max_cells: int | None = None,
        seed: int | None = None,
    ) -> dict[str, np.ndarray | torch.Tensor]:
        """
        Retrieve the current sketch (non-destructively).

        Returns:
            A dict with:

            * ``"obs_names"`` — ``np.ndarray`` of cell IDs, always present.
            * ``"x_ng"`` — sparse CSR tensor of shape ``(N_sketch, G)``,
              present when ``return_cell_data=True``.
        """

    @abstractmethod
    def apply_bucket_filters(
        self,
        min_cells_per_bucket: int = 1,
        min_metadata_diversity: int = 1,
    ) -> None:
        """Prune buckets in-place that fail density or diversity thresholds."""

    @property
    @abstractmethod
    def total_cells(self) -> int:
        """Total cells currently stored in the sketch."""

    @property
    @abstractmethod
    def num_filled_buckets(self) -> int:
        """Number of buckets containing at least one cell."""

    @abstractmethod
    def _reset_storage(self) -> None:
        """Discard all accumulated sketch data."""

    @abstractmethod
    def _storage_state(self) -> dict[str, Any]:
        """The sketch data to put in a checkpoint."""

    @abstractmethod
    def _load_storage_state(self, state: dict[str, Any]) -> None:
        """Restore the sketch data from a checkpoint's ``sketch_state``."""

    def _final_reservoir_kwargs(self) -> dict[str, Any]:
        """Arguments of the :meth:`get_reservoir` call that stores ``sketch_obs_names`` at the end of training."""
        return {}

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def on_train_start(self, trainer: pl.Trainer) -> None:
        if trainer.world_size > 1:
            raise RuntimeError(
                f"{self.__class__.__name__} only supports single-device training "
                f"(got world_size={trainer.world_size}). Run on a single GPU or CPU."
            )

    def on_train_batch_end(self, trainer: pl.Trainer) -> None:
        self._batches_seen += 1

        total = self.total_cells
        # Clamped at 0: the only source of a negative delta during training is a coarsening
        # event in StreamingPlaidGeometricSketch, which shrinks total_cells in a single batch.
        # Letting that spike into the EMA would contaminate proj_total_cells (a display-only
        # projection) with several batches of spuriously negative values.
        delta = max(total - self._prev_total_cells, 0)
        self._ema_delta = 0.2 * delta + 0.8 * self._ema_delta
        self._prev_total_cells = total

        assert isinstance(trainer.model, pl.LightningModule)
        trainer.model.log("current_cells", float(total), prog_bar=True)

        n_total = trainer.num_training_batches
        if n_total != float("inf"):
            batches_remaining = n_total - self._batches_seen
            projected = total + self._ema_delta * batches_remaining
            trainer.model.log("proj_total_cells", projected, prog_bar=True)

    def on_train_epoch_end(self, trainer: pl.Trainer) -> None:
        trainer.should_stop = True
        self.apply_bucket_filters(self.min_cells_per_bucket, self.min_metadata_diversity)
        self.sketch_obs_names = self.get_reservoir(**self._final_reservoir_kwargs())["obs_names"]
        assert isinstance(trainer.model, pl.LightningModule)
        trainer.model.log("final_sketch_size", float(len(self.sketch_obs_names)))

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["sketch_state"] = {
            **self._storage_state(),
            "batches_seen": self._batches_seen,
            "prev_total_cells": self._prev_total_cells,
            "ema_delta": self._ema_delta,
            "sketch_obs_names": self.sketch_obs_names,
        }

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        state = checkpoint["sketch_state"]
        self._load_storage_state(state)
        self._batches_seen = state["batches_seen"]
        self._prev_total_cells = state["prev_total_cells"]
        self._ema_delta = state["ema_delta"]
        self.sketch_obs_names = state["sketch_obs_names"]

    # ------------------------------------------------------------------
    # Parameter reset
    # ------------------------------------------------------------------

    def reset_parameters(self) -> None:
        self._reset_storage()
        self._batches_seen = 0
        self._prev_total_cells = 0
        self._ema_delta = 0.0
        self.sketch_obs_names = np.empty(0, dtype=object)
        self._generator.manual_seed(self._seed)
        self._dummy_param.data.zero_()


class StreamingHyperplaneGeometricSketch(StreamingGeometricSketch):
    """
    Online geometric sketching via locality-sensitive hashing (LSH).

    Streams single-cell gene expression data and retains a geometrically diverse
    sketch of cells across a single pass. Rare cell states are preserved while
    over-represented dense clusters are capped at ``max_cells_per_bucket``.

    Cells are bucketed by projecting into a low-dimensional space and binarizing
    the result to produce an integer bucket ID. Each bucket maintains a reservoir
    of up to ``max_cells_per_bucket`` cells via uniform reservoir sampling.

    By default the projection is a frozen random Gaussian linear map from gene
    space to ``n_bits`` dimensions (classical random-hyperplane LSH). Optionally
    a pre-trained module (e.g. a frozen PCA or scVI encoder) can be provided as
    ``projector``; its output is then passed through the LSH random linear layer,
    so ``n_bits`` controls bucket granularity independently of the encoder's output
    dimension.

    Only single-device training is supported. A ``RuntimeError`` is raised at the
    start of training if more than one device is detected.

    Args:
        var_names_g:
            Gene names for input validation.
        n_bits:
            Number of LSH projection bits. Determines up to ``2^n_bits`` buckets.
        max_cells_per_bucket:
            Maximum cells retained per bucket via reservoir sampling.
        min_cells_per_bucket:
            Density threshold applied at the end of training. Buckets that
            observed fewer cells in total are dropped. Default of ``1``
            disables filtering.
        min_metadata_diversity:
            Diversity threshold applied at the end of training. Buckets that
            observed fewer unique metadata categories are dropped. Default of
            ``1`` disables filtering.
        store_cell_data:
            If ``True`` (default), accumulate sparse cell expression vectors.
            If ``False``, only cell IDs (obs_names) are stored; calling
            ``get_reservoir(return_cell_data=True)`` will raise.
        projector:
            Optional frozen encoder mapping ``(N, G) → (N, D)``. When given, its
            output feeds the LSH linear layer rather than raw gene expression.
            The module's gradients are disabled on assignment.
        limit_input_to_top_pcs:
            If specified, limit the input to this number of top principal components.
        seed:
            Random seed for the LSH projection weights and for reservoir sampling.
            Sampling draws from a generator owned by this model, so results are
            unaffected by other consumers of the global :mod:`torch` RNG.
    """

    # The sketch state, (re)initialized by _reset_storage().
    _bucket_cells: dict[tuple[int, ...], list[torch.Tensor]]
    _bucket_obs_names: dict[tuple[int, ...], list[str]]
    _bucket_total_seen: dict[tuple[int, ...], int]
    _bucket_metadata: dict[tuple[int, ...], set[int]]

    def __init__(
        self,
        var_names_g: np.ndarray,
        n_bits: int = 12,
        max_cells_per_bucket: int = 100,
        min_cells_per_bucket: int = 1,
        min_metadata_diversity: int = 1,
        store_cell_data: bool = True,
        projector: nn.Module | None = None,
        limit_input_to_top_pcs: int | None = None,
        seed: int = 0,
    ) -> None:
        # Set before super().__init__() so reset_parameters() can reference them.
        self.n_bits = n_bits
        self.num_buckets = 2**n_bits
        super().__init__(
            var_names_g=var_names_g,
            max_cells_per_bucket=max_cells_per_bucket,
            min_cells_per_bucket=min_cells_per_bucket,
            min_metadata_diversity=min_metadata_diversity,
            store_cell_data=store_cell_data,
            projector=projector,
            limit_input_to_top_pcs=limit_input_to_top_pcs,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Lazy initialization
    # ------------------------------------------------------------------

    def _lazy_init(self, x_ng: torch.Tensor) -> None:
        """Build the LSH layer on the first forward call."""
        if hasattr(self, "lsh_layer"):
            return

        device = x_ng.device

        if self.projector is not None:
            with torch.no_grad():
                sample = x_ng[:1].float()
                if sample.layout != torch.strided:
                    sample = sample.to_dense()
                out = self.projector(sample)
            D = out.shape[1]
        else:
            D = len(self.var_names_g)

        self.lsh_layer = nn.Linear(D, self.n_bits, bias=False).to(device)
        with torch.no_grad():
            # Initialize on CPU for determinism across devices, then copy.
            gen = torch.Generator().manual_seed(self._seed)
            w = torch.empty(self.n_bits, D, dtype=torch.float32)
            nn.init.normal_(w, generator=gen)
            w /= w.norm(dim=1, keepdim=True).clamp(min=1e-8)
            self.lsh_layer.weight.data.copy_(w)
        self.lsh_layer.requires_grad_(False)

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_bucket_ids(self, x_ng: torch.Tensor) -> torch.Tensor:
        """Project ``x_ng`` through the encoder (if any) and LSH layer → integer bucket IDs."""
        x = x_ng.float()
        if self.projector is not None:
            x = self.projector(x)
        logits = self.lsh_layer(x)  # (N, n_bits)
        bits = (logits > 0).long()
        powers = 2 ** torch.arange(self.n_bits, device=x.device, dtype=torch.long)
        return bits @ powers  # (N,)

    def forward(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
        obs_names_n: np.ndarray,
        metadata_n: np.ndarray | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """
        Accumulate a minibatch of cells into the geometric sketch reservoir.

        Args:
            x_ng:
                Expression matrix of shape ``(N, G)``, dense or sparse.
            var_names_g:
                Gene names for the batch; must match ``self.var_names_g``.
            obs_names_n:
                Cell IDs for the batch, shape ``(N,)``.
            metadata_n:
                Optional integer metadata of shape ``(N,)`` for diversity
                tracking.  When provided, each cell's value is recorded in
                its bucket's metadata set, enabling :meth:`apply_bucket_filters`
                with ``min_metadata_diversity > 1``.

        Returns:
            An empty dict (no loss is computed).
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "self.var_names_g", self.var_names_g)

        if self.limit_input_to_top_pcs is not None:
            x_ng = x_ng[:, : self.limit_input_to_top_pcs]

        self._lazy_init(x_ng)
        self.update(x_ng, obs_names_n, metadata_n)
        return {}

    @torch.no_grad()
    def update(
        self,
        x_ng: torch.Tensor,
        obs_names_n: np.ndarray,
        metadata_n: np.ndarray | None = None,
    ) -> int:
        """
        Update the reservoir with a minibatch of cells.

        Args:
            x_ng:
                Expression matrix of shape ``(N, G)``, dense or sparse.
            obs_names_n:
                Cell IDs of shape ``(N,)``.
            metadata_n:
                Optional integer metadata of shape ``(N,)``.

        Returns:
            Number of cells inserted or replaced in this update.
        """
        x_float = x_ng.float()
        x_dense = x_float.to_dense() if x_float.layout != torch.strided else x_float

        bucket_ids = self._compute_bucket_ids(x_dense)
        inserted_count = 0

        for b_id in torch.unique(bucket_ids):
            b_key = (int(b_id.item()),)
            cell_indices = (bucket_ids == b_id).nonzero(as_tuple=True)[0]

            if b_key not in self._bucket_total_seen:
                self._bucket_total_seen[b_key] = 0
                self._bucket_obs_names[b_key] = []
                self._bucket_metadata[b_key] = set()
                if self.store_cell_data:
                    self._bucket_cells[b_key] = []

            for idx in cell_indices:
                i = int(idx.item())
                seen = self._bucket_total_seen[b_key]
                count = len(self._bucket_obs_names[b_key])
                self._bucket_total_seen[b_key] += 1
                obs_name = str(obs_names_n[i])

                if metadata_n is not None:
                    self._bucket_metadata[b_key].add(int(metadata_n[i]))

                if count < self.max_cells_per_bucket:
                    self._bucket_obs_names[b_key].append(obs_name)
                    if self.store_cell_data:
                        self._bucket_cells[b_key].append(x_dense[i].to_sparse())
                    inserted_count += 1
                else:
                    r = int(torch.randint(0, seen + 1, (1,), generator=self._generator).item())
                    if r < self.max_cells_per_bucket:
                        self._bucket_obs_names[b_key][r] = obs_name
                        if self.store_cell_data:
                            self._bucket_cells[b_key][r] = x_dense[i].to_sparse()
                        inserted_count += 1

        return inserted_count

    # ------------------------------------------------------------------
    # Retrieval and filtering
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_reservoir(
        self,
        return_cell_data: bool = False,
        max_cells: int | None = None,
        seed: int | None = None,
    ) -> dict[str, np.ndarray | torch.Tensor]:
        """
        Retrieve the current sketch.

        Buckets are visited in sorted key order so the pre-downsample
        ordering is deterministic.  This method is non-destructive; call
        :meth:`apply_bucket_filters` separately to prune the internal state.

        Args:
            return_cell_data:
                If ``True``, include sparse cell expression in the output.
                Requires ``store_cell_data=True`` at construction.
            max_cells:
                If given, randomly downsample the result to at most this
                many cells.  When the reservoir is already smaller than
                ``max_cells`` no downsampling occurs.
            seed:
                Random seed for the ``max_cells`` downsample.  ``None``
                gives a non-reproducible draw.

        Returns:
            A dict with:

            * ``"obs_names"`` — ``np.ndarray`` of cell IDs, always present.
            * ``"x_ng"`` — sparse CSR tensor of shape ``(N_sketch, G)``,
              present when ``return_cell_data=True``.
        """
        if return_cell_data and not self.store_cell_data:
            raise ValueError("store_cell_data=False was set at construction; cell expression data was not accumulated.")

        all_obs: list[str] = []
        all_cells: list[torch.Tensor] = []

        for b_key in sorted(self._bucket_obs_names):
            all_obs.extend(self._bucket_obs_names[b_key])
            if return_cell_data:
                all_cells.extend(self._bucket_cells[b_key])

        if max_cells is not None and len(all_obs) > max_cells:
            rng = np.random.default_rng(seed)
            indices = np.sort(rng.choice(len(all_obs), size=max_cells, replace=False))
            all_obs = [all_obs[i] for i in indices]
            if return_cell_data:
                all_cells = [all_cells[i] for i in indices]

        result: dict[str, np.ndarray | torch.Tensor] = {"obs_names": np.array(all_obs)}

        if return_cell_data:
            if all_cells:
                result["x_ng"] = torch.cat([c.unsqueeze(0) for c in all_cells], dim=0).to_sparse_csr()
            else:
                result["x_ng"] = torch.zeros(0, len(self.var_names_g)).to_sparse_csr()

        return result

    def apply_bucket_filters(
        self,
        min_cells_per_bucket: int = 1,
        min_metadata_diversity: int = 1,
    ) -> None:
        """
        Prune buckets in-place that fail density or diversity thresholds.

        Uses ``_bucket_total_seen`` (the total number of cells ever hashed
        to a bucket, not the capped retained count) as the density measure.
        This correctly identifies sparse regions even when ``max_cells_per_bucket``
        caps the retained count below the threshold.

        Args:
            min_cells_per_bucket:
                Buckets where fewer than this many cells were ever observed
                are deleted.
            min_metadata_diversity:
                Buckets that observed fewer than this many unique metadata
                categories (from ``metadata_n``) are deleted.  Ignored
                when ``min_metadata_diversity <= 1``.

        Raises:
            ValueError:
                If ``min_metadata_diversity > 1`` but no metadata was ever
                tracked (i.e. ``metadata_n`` was never passed to
                :meth:`update` or :meth:`forward`).
        """
        if (
            min_metadata_diversity > 1
            and self._bucket_metadata
            and all(len(v) == 0 for v in self._bucket_metadata.values())
        ):
            raise ValueError(
                "min_metadata_diversity > 1 but no metadata was tracked. Pass metadata_n to forward() or update()."
            )

        keys_to_delete = []
        for k, seen in self._bucket_total_seen.items():
            if seen < min_cells_per_bucket:
                keys_to_delete.append(k)
                continue
            if min_metadata_diversity > 1 and len(self._bucket_metadata[k]) < min_metadata_diversity:
                keys_to_delete.append(k)

        for k in keys_to_delete:
            del self._bucket_total_seen[k]
            del self._bucket_obs_names[k]
            del self._bucket_metadata[k]
            if self.store_cell_data:
                del self._bucket_cells[k]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_cells(self) -> int:
        """Total cells currently stored in the sketch."""
        return sum(len(v) for v in self._bucket_obs_names.values())

    @property
    def num_filled_buckets(self) -> int:
        """Number of buckets containing at least one cell."""
        return len(self._bucket_obs_names)

    @property
    def bucket_fill_fraction(self) -> float:
        """Fraction of the ``2^n_bits`` buckets that are occupied."""
        return self.num_filled_buckets / self.num_buckets

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def on_train_batch_end(self, trainer: pl.Trainer) -> None:
        super().on_train_batch_end(trainer)
        assert isinstance(trainer.model, pl.LightningModule)
        trainer.model.log("fill_frac", self.bucket_fill_fraction, prog_bar=True)

    def _storage_state(self) -> dict[str, Any]:
        return {
            "bucket_obs_names": self._bucket_obs_names,
            "bucket_cells": self._bucket_cells,
            "bucket_total_seen": self._bucket_total_seen,
            "bucket_metadata": self._bucket_metadata,
        }

    def _load_storage_state(self, state: dict[str, Any]) -> None:
        self._bucket_obs_names = state["bucket_obs_names"]
        self._bucket_cells = state["bucket_cells"]
        self._bucket_total_seen = state["bucket_total_seen"]
        self._bucket_metadata = state["bucket_metadata"]

    # ------------------------------------------------------------------
    # Parameter reset
    # ------------------------------------------------------------------

    def _reset_storage(self) -> None:
        self._bucket_cells = {}
        self._bucket_obs_names = {}
        self._bucket_total_seen = {}
        self._bucket_metadata = {}

    def reset_parameters(self) -> None:
        """
        Reset all accumulated sketch data and re-seed the LSH projection weights.

        Safe to call before or after lazy initialization.
        """
        super().reset_parameters()
        if hasattr(self, "lsh_layer"):
            with torch.no_grad():
                D = self.lsh_layer.weight.shape[1]
                gen = torch.Generator().manual_seed(self._seed)
                w = torch.empty(self.n_bits, D, dtype=torch.float32)
                nn.init.normal_(w, generator=gen)
                w /= w.norm(dim=1, keepdim=True).clamp(min=1e-8)
                self.lsh_layer.weight.data.copy_(w)
            self.lsh_layer.requires_grad_(False)


class StreamingPlaidGeometricSketch(StreamingGeometricSketch):
    r"""
    Online geometric sketching [1] via Dynamic Spatial Hashing.

    Streams single-cell gene expression data and retains a geometrically diverse
    sketch of cells across a single pass. It maintains a plaid $\epsilon$-cover of the
    embedding space: an axis-aligned grid whose occupied voxels each hold a reservoir of
    up to ``max_cells_per_bucket`` cells. Whenever the number of occupied voxels exceeds
    ``target_voxels`` the grid is coarsened and colliding voxels are merged exactly (no
    per-cell coordinates are stored). With ``max_cells_per_bucket=1`` this is the sketch of [1]:
    one cell per occupied box. With a few cells per voxel, the final sketch can be filled up to
    exactly ``target_n_cells`` as in [1] (see below).

    To combat technical artifacts, an end-of-epoch pruning step removes voxels
    that fail to meet minimum cell count (density) or minimum categorical diversity
    (e.g., number of unique datasets or patients) thresholds.

    **Grid geometry.** As in the reference implementation of [1], the grid lives in the
    embedding's *native units*: every axis starts with the same scalar voxel size (axes are
    not rescaled or whitened, so Euclidean distances in the embedding keep their meaning),
    and an axis whose spread is smaller than the voxel size is effectively never split. The
    effective dimensionality of the cover therefore adapts to the data instead of growing
    with the number of embedding dimensions. The starting voxel size is a small fraction of
    the largest per-axis standard deviation of the first minibatch, so the grid starts finer
    than needed: coarsening is irreversible, starting too coarse is not recoverable.

    **Grid origin.** The origin is estimated once from the first minibatch (just below its
    per-axis minimum) and then held fixed. A permanent grid boundary sits at the origin for
    every voxel size, so an origin at zero would bisect zero-centered embeddings (e.g. PCA
    output) down the middle on every axis and prevent coarsening from ever merging those cells.

    **Coarsening.** The box length of [1] is a continuous scalar found by bisection on a static
    dataset. Here the grid can only double its spacing (so that old voxels nest exactly inside
    new ones), and it does so **one axis at a time**, always the axis with the currently smallest
    voxel size (so axes take turns and stay within a factor of 2 of each other). Doubling a single
    axis merges voxels in pairs at most, so the occupied voxel count falls by at most a factor of 2
    per step, in any dimension. A coarsening event takes the smallest number of such steps that
    brings the count to ``target_voxels`` or fewer, so it lands in ``(target_voxels / 2, target_voxels]``.
    (Doubling all axes at once can collapse the cover from far above ``target_voxels`` to a single voxel
    in one step in moderate-to-high dimension, and that cannot be undone.) The steps of an event are
    not carried out one by one: the number of steps is found by bisection, with each candidate
    evaluated by hashing the voxel coordinates shifted right by the per-axis number of doublings,
    and the voxels are then merged in a single pass.

    **Storage.** All state is a handful of tensors on the device of the embedding (a GPU, if
    training on one): a row per occupied voxel with its integer coordinates, a 64-bit hash of them
    (voxels are identified by the hash), the number of cells seen, and the retained cells. Cells are
    retained by *random tags*: every cell draws an independent uniform tag and a voxel keeps the
    ``max_cells_per_bucket`` cells with the smallest tags, which is a uniform reservoir sample and
    merges exactly when voxels merge. Inserting a minibatch and coarsening are sort / scatter
    kernels with no Python loop over voxels or cells. Cell names (and cell data, if
    ``store_cell_data``) live host-side, keyed by integer cell ids.

    **Exactly ``target_n_cells`` cells.** The voxel count alone does not give the cell count (the
    count after a coarsening event is anywhere in ``(target_voxels / 2, target_voxels]``). If
    ``target_n_cells`` is given, :meth:`get_reservoir` (with ``max_cells`` as that number, which is also
    what is stored in ``sketch_obs_names`` at the end of training) does what [1] does at the end:
    it coarsens a copy of the grid to the coarsest nested grid that still has at least
    ``target_n_cells`` occupied voxels, and takes one cell from each of ``target_n_cells`` randomly
    chosen voxels. If the grid is already too coarse for that (fewer voxels than ``target_n_cells``)
    it instead cycles through the voxels, round by round, taking each voxel's next retained cell,
    which needs ``max_cells_per_bucket > 1``. A good choice is ``target_voxels`` of about
    ``2 * target_n_cells`` and ``max_cells_per_bucket=2``, so that the grid has room to be coarsened
    into place at the end.

    Only single-device training is supported. A ``RuntimeError`` is raised at the
    start of training if more than one device is detected.

    References:
        [1] Hie, B., ..., Berger, B. (2019). Geometric sketching compactly summarizes
            the single-cell transcriptomic landscape. Cell Systems, 8(6), 483-493.e7.

    Args:
        var_names_g:
            Gene names for input validation.
        target_voxels:
            Maximum number of occupied voxels. The grid coarsens whenever this is exceeded.
        max_cells_per_bucket:
            Maximum cells retained per voxel via uniform reservoir sampling. The retained cells take
            16 bytes per slot of every voxel on the device, so keep this small (1 or 2).
        min_cells_per_bucket:
            Density threshold applied at the end of training via
            :meth:`apply_bucket_filters`. Voxels that observed fewer cells
            in total are dropped. Default of ``1`` disables filtering.
        min_metadata_diversity:
            Diversity threshold applied at the end of training via
            :meth:`apply_bucket_filters`. Voxels that observed fewer unique
            metadata categories (from ``metadata_n``) are dropped.
        store_cell_data:
            If ``True``, accumulate sparse cell expression vectors.
        projector:
            Optional frozen encoder mapping ``(N, G) → (N, D)``.
        limit_input_to_top_pcs:
            If specified, limit the input to this number of top principal components.
        seed:
            Random seed for reservoir sampling and for the final selection of cells. Sampling
            draws from a generator owned by this model, so results are unaffected
            by other consumers of the global :mod:`torch` RNG.
        target_n_cells:
            If given, the exact size of the sketch stored in ``sketch_obs_names`` at the end of
            training (when the sketch has at least that many cells). Must not exceed
            ``target_voxels * max_cells_per_bucket``.
    """

    # The starting voxel size, on every axis, is this fraction of the largest per-axis standard
    # deviation of the first minibatch (see _lazy_init). Deliberately small: coarsening is
    # recoverable, starting too coarse is not.
    _INITIAL_VOXEL_STD_FRACTION = 0.01

    # The grid origin sits this many per-axis standard deviations below the first minibatch's
    # per-axis minimum, so that (almost) no later cell gets a negative voxel coordinate: floor
    # division by 2 keeps negative coordinates apart from non-negative ones forever, so such
    # cells could never merge with the bulk however coarse the grid gets.
    _ORIGIN_MARGIN_STDS = 5.0

    # The cell pool is compacted (cells that lost their slot are dropped) when it holds this many
    # times the number of live cells, plus a constant.
    _POOL_SLACK_FACTOR = 2
    _POOL_SLACK_CELLS = 65536

    # The sketch state, (re)initialized by _reset_storage().
    _store: _VoxelStore | None
    _pool: _CellPool
    _next_id: int
    _max_batch: int
    _coarsen_event_count: int
    _coarsening_exhausted: bool
    _metadata_values: list[int]
    _metadata_code_of: dict[int, int]

    def __init__(
        self,
        var_names_g: np.ndarray,
        target_voxels: int = 100_000,
        max_cells_per_bucket: int = 1,
        min_cells_per_bucket: int = 1,
        min_metadata_diversity: int = 1,
        store_cell_data: bool = False,
        projector: nn.Module | None = None,
        limit_input_to_top_pcs: int | None = None,
        seed: int = 0,
        target_n_cells: int | None = None,
    ) -> None:
        if target_n_cells is not None and target_n_cells > target_voxels * max_cells_per_bucket:
            raise ValueError(
                f"target_n_cells={target_n_cells} can never be reached: the sketch holds at most "
                f"target_voxels * max_cells_per_bucket = {target_voxels * max_cells_per_bucket} cells."
            )
        # Set before super().__init__() so reset_parameters() can reference them.
        self.target_voxels = target_voxels
        self.target_n_cells = target_n_cells

        super().__init__(
            var_names_g=var_names_g,
            max_cells_per_bucket=max_cells_per_bucket,
            min_cells_per_bucket=min_cells_per_bucket,
            min_metadata_diversity=min_metadata_diversity,
            store_cell_data=store_cell_data,
            projector=projector,
            limit_input_to_top_pcs=limit_input_to_top_pcs,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Lazy initialization
    # ------------------------------------------------------------------

    @staticmethod
    def _state_device(z: torch.Tensor) -> torch.device:
        # The sort / scatter kernels the sketch relies on are not all implemented on MPS.
        return torch.device("cpu") if z.device.type == "mps" else z.device

    @torch.no_grad()
    def _lazy_init(self, z: torch.Tensor) -> None:
        """Estimate a fixed grid origin and the starting voxel_size from the first minibatch seen.

        Origin: a grid boundary always sits at ``voxel_offset`` itself, for every ``voxel_size``
        (``floor((offset - offset) / voxel_size) == 0`` regardless of scale). Left at the
        default of zero, that permanent boundary runs straight through the densest part of
        zero-centered embeddings (e.g. PCA output), so growing ``voxel_size`` can never merge
        cells that straddle it. The origin is instead placed ``_ORIGIN_MARGIN_STDS`` standard
        deviations below the first minibatch's per-axis minimum, so that (almost) every cell
        has a non-negative coordinate and everything eventually collapses to one side of it.

        Starting voxel_size: the same scalar on every axis (native units, as in the reference
        implementation of geometric sketching), a small fraction of the largest per-axis
        standard deviation. It deliberately undershoots the final voxel size, since coarsening
        is cheap and starting too coarse is unrecoverable.
        """
        if not hasattr(self, "voxel_offset"):
            # unbiased=False avoids NaN when the first minibatch has a single cell (N=1).
            std = z.std(dim=0, unbiased=False)
            offset = z.min(dim=0).values - self._ORIGIN_MARGIN_STDS * std
            self.register_buffer("voxel_offset", offset)

            scale = std.max()
            if not scale > 0:  # degenerate first minibatch (a single cell, or identical cells)
                scale = torch.ones_like(scale)
            voxel_size = torch.full_like(std, self._INITIAL_VOXEL_STD_FRACTION * float(scale))
            self.register_buffer("voxel_size", voxel_size)

        if self._store is None:
            self._store = _VoxelStore(z.shape[1], self.max_cells_per_bucket, self._seed, self._state_device(z))

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    def _metadata_codes(self, metadata_n: np.ndarray, device: torch.device) -> torch.Tensor:
        """Map the metadata categories of a batch to consecutive integer codes (assigned in order of appearance)."""
        values = np.asarray(metadata_n).astype(np.int64).reshape(-1)
        unique, inverse = np.unique(values, return_inverse=True)
        codes = np.empty(len(unique), dtype=np.int64)
        for j, value in enumerate(unique.tolist()):
            if value not in self._metadata_code_of:
                self._metadata_code_of[value] = len(self._metadata_values)
                self._metadata_values.append(value)
            codes[j] = self._metadata_code_of[value]
        return torch.as_tensor(codes[inverse.reshape(-1)], device=device)

    @torch.no_grad()
    def _compact_pool(self) -> None:
        """Drop the cells that no longer hold a slot from the cell pool."""
        store = self._store
        assert store is not None
        live = store.slot_id[: store.n]
        live_np = live[live >= 0].cpu().numpy()
        live_np.sort()
        self._pool.compact(live_np)

    @torch.no_grad()
    def _coarsen(self) -> None:
        """Coarsen the grid, one axis at a time, until at most ``target_voxels`` voxels are occupied.

        Each step doubles the voxel size along the axis with the currently smallest voxel size
        (ties go to the lowest axis index), so axes take turns and their voxel sizes stay within
        a factor of 2 of one another. A single-axis step cannot reduce the occupied voxel count
        by more than a factor of 2, so stopping at the first step that gets to ``target_voxels``
        or fewer leaves more than ``target_voxels / 2`` voxels, regardless of the dimension.
        (Doubling all axes at once can drop the count from far above ``target_voxels`` to a single
        voxel in moderate-to-high dimension.)

        The steps are not performed one at a time. The sequence of axes is known up front, the
        occupied voxel count is monotone in the number of steps, so the number of steps is found
        by galloping bisection (each candidate costs a hash of the shifted coordinates and a sort),
        and the voxels are merged once.

        If ``target_voxels`` is unreachable (a cell that falls below the grid origin has a
        negative coordinate, which is never merged with a non-negative one) the grid is coarsened
        as far as it goes, with a warning.
        """
        store = self._store
        assert store is not None
        self._coarsen_event_count += 1
        order = store.coarsening_order(self.voxel_size)
        steps = _first_true(
            lambda s: store.count_after(store.shifts_after(order, s)) <= self.target_voxels, order.numel()
        )
        if steps > order.numel():
            warnings.warn(
                f"The grid cannot be coarsened to {self.target_voxels} voxels (it still has more after every "
                "axis has been coarsened as far as it goes). This happens if many cells fall below the grid "
                "origin estimated from the first minibatch. Coarsening is not attempted again.",
                stacklevel=2,
            )
            steps = order.numel()
            self._coarsening_exhausted = True
        shifts = store.shifts_after(order, steps)
        store.merge_to(shifts, capacity=self.target_voxels + self._max_batch)
        self.voxel_size.mul_(torch.exp2(shifts.to(self.voxel_size)))
        self._compact_pool()

    def forward(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
        obs_names_n: np.ndarray,
        metadata_n: np.ndarray | None = None,
    ) -> dict[str, torch.Tensor | None]:
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "self.var_names_g", self.var_names_g)

        if self.min_metadata_diversity > 1 and metadata_n is None:
            raise ValueError("metadata_n must be provided when min_metadata_diversity > 1.")

        if self.limit_input_to_top_pcs is not None:
            x_ng = x_ng[:, : self.limit_input_to_top_pcs]

        self.update(x_ng, obs_names_n, metadata_n)
        return {}

    @torch.no_grad()
    def update(
        self,
        x_ng: torch.Tensor,
        obs_names_n: np.ndarray,
        metadata_n: np.ndarray | None = None,
    ) -> int:
        x_float = x_ng.float()
        x_dense = x_float.to_dense() if x_float.layout != torch.strided else x_float

        if self.projector is not None:
            z = self.projector(x_dense)
        else:
            z = x_dense

        batch = z.shape[0]
        if batch == 0:
            return 0

        self._lazy_init(z)
        store = self._store
        assert store is not None
        device = self._state_device(z)
        store.to(device)

        coords = torch.floor((z - self.voxel_offset) / self.voxel_size).long().to(device)
        # Tags are drawn on the host from the model's generator, so results do not depend on the device.
        tags = torch.rand(batch, generator=self._generator, dtype=torch.float64).to(device)
        codes = None if metadata_n is None else self._metadata_codes(metadata_n, device)
        self._max_batch = max(self._max_batch, batch)

        won = store.insert(coords, tags, self._next_id, codes, self.target_voxels + self._max_batch)
        won_np = np.sort(won.cpu().numpy())
        if len(won_np):
            names = np.empty(len(won_np), dtype=object)
            names[:] = [str(obs_names_n[i]) for i in won_np]
            rows = None
            if self.store_cell_data:
                rows = _to_scipy_csr(x_dense[torch.as_tensor(won_np, device=x_dense.device)])
            self._pool.append(self._next_id + won_np, names, rows)
        self._next_id += batch

        if len(self._pool) > self._POOL_SLACK_FACTOR * store.n * store.slots + self._POOL_SLACK_CELLS:
            self._compact_pool()
        if store.n > self.target_voxels and not self._coarsening_exhausted:
            self._coarsen()

        return len(won_np)

    # ------------------------------------------------------------------
    # Retrieval and filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _all_cell_ids(store: _VoxelStore) -> torch.Tensor:
        """Ids of all retained cells, ordered by voxel (hash) and then by rank within the voxel."""
        order = torch.argsort(store.hashes[: store.n])
        ids = store.slot_id[: store.n][order].reshape(-1)
        return ids[ids >= 0]

    @torch.no_grad()
    def _coarsest_with_at_least(self, n_voxels: int) -> _VoxelStore:
        """A copy of the store, coarsened to the coarsest nested grid that still has ``n_voxels`` or more voxels."""
        store = self._store
        assert store is not None
        if store.n <= n_voxels:
            return store
        order = store.coarsening_order(self.voxel_size)
        first_too_coarse = _first_true(
            lambda s: store.count_after(store.shifts_after(order, s)) < n_voxels, order.numel()
        )
        steps = first_too_coarse - 1
        return store if steps == 0 else store.coarsened(store.shifts_after(order, steps))

    @staticmethod
    def _round_robin_cell_ids(store: _VoxelStore, n_cells: int, generator: torch.Generator) -> torch.Tensor:
        """
        Choose ``n_cells`` cells as in the geometric sketching paper: one cell from each voxel (round 0, from
        a random subset of the voxels if there are more than ``n_cells`` of them), then, only if cells are
        still missing, each voxel's next retained cell (round 1, from a random subset of the voxels that have
        one), and so on.
        """
        order = torch.argsort(store.hashes[: store.n])
        slot_id = store.slot_id[: store.n][order]
        chosen: list[torch.Tensor] = []
        taken = 0
        for rank in range(store.slots):
            ids = slot_id[:, rank]
            ids = ids[ids >= 0]
            if taken + ids.numel() > n_cells:
                pick = torch.randperm(ids.numel(), generator=generator)[: n_cells - taken].sort().values
                ids = ids[pick.to(ids.device)]
            chosen.append(ids)
            taken += ids.numel()
            if taken >= n_cells:
                break
        return torch.cat(chosen)

    @torch.no_grad()
    def get_reservoir(
        self,
        return_cell_data: bool = False,
        max_cells: int | None = None,
        seed: int | None = None,
    ) -> dict[str, np.ndarray | torch.Tensor]:
        """
        Retrieve the current sketch.

        This method is non-destructive; call :meth:`apply_bucket_filters` separately to prune the internal
        state. Cells are ordered by voxel (in the order of the voxels' hashes, which is deterministic).

        Args:
            return_cell_data:
                If ``True``, include sparse cell expression in the output.
                Requires ``store_cell_data=True`` at construction.
            max_cells:
                If given and the sketch holds more cells than this, return exactly this many cells,
                spread evenly over the occupied regions as in the geometric sketching paper: coarsen a
                copy of the grid to the coarsest nested grid that still has at least ``max_cells``
                voxels and take the retained cell of ``max_cells`` randomly chosen voxels. If the grid
                has fewer voxels than ``max_cells`` already, take every voxel's first retained cell, then
                the second retained cell of randomly chosen voxels, and so on. A sketch with no more than
                ``max_cells`` cells is returned whole.
            seed:
                Random seed for the choice of voxels. ``None`` uses the model's own ``seed``, so
                repeated calls give the same cells.

        Returns:
            A dict with:

            * ``"obs_names"`` — ``np.ndarray`` of cell IDs, always present.
            * ``"x_ng"`` — sparse CSR tensor of shape ``(N_sketch, G)``,
              present when ``return_cell_data=True``.
        """
        if return_cell_data and not self.store_cell_data:
            raise ValueError("store_cell_data=False was set at construction; cell expression data was not accumulated.")

        store = self._store
        ids = np.empty(0, dtype=np.int64)
        if store is not None and store.n > 0:
            if max_cells is not None and store.total_cells() > max_cells:
                generator = torch.Generator().manual_seed(self._seed if seed is None else seed)
                coarse = self._coarsest_with_at_least(max_cells)
                ids = self._round_robin_cell_ids(coarse, max_cells, generator).cpu().numpy()
            else:
                ids = self._all_cell_ids(store).cpu().numpy()

        names, rows = self._pool.take(ids)
        result: dict[str, np.ndarray | torch.Tensor] = {"obs_names": np.asarray(names, dtype=str)}

        if return_cell_data:
            if rows is None or rows.shape[0] == 0:
                result["x_ng"] = torch.zeros(0, len(self.var_names_g)).to_sparse_csr()
            else:
                result["x_ng"] = torch.sparse_csr_tensor(
                    torch.from_numpy(rows.indptr.astype(np.int64)),
                    torch.from_numpy(rows.indices.astype(np.int64)),
                    torch.from_numpy(rows.data),
                    size=tuple(rows.shape),
                )

        return result

    @torch.no_grad()
    def apply_bucket_filters(
        self,
        min_cells_per_bucket: int = 1,
        min_metadata_diversity: int = 1,
    ) -> None:
        """
        Prune voxels in-place that fail density or diversity thresholds.

        Uses the total number of cells ever assigned to a voxel (not the capped retained count)
        as the density measure. This correctly identifies sparse regions even when
        ``max_cells_per_bucket`` caps the retained count below the threshold.

        Args:
            min_cells_per_bucket:
                Voxels where fewer than this many cells were ever observed
                are deleted.
            min_metadata_diversity:
                Voxels that observed fewer than this many unique metadata
                categories (from ``metadata_n``) are deleted.  Ignored
                when ``min_metadata_diversity <= 1``.

        Raises:
            ValueError:
                If ``min_metadata_diversity > 1`` but no metadata was ever
                tracked (i.e. ``metadata_n`` was never passed to
                :meth:`update` or :meth:`forward`).
        """
        store = self._store
        if store is None or store.n == 0:
            return
        if min_metadata_diversity > 1 and not store.has_metadata:
            raise ValueError(
                "min_metadata_diversity > 1 but no metadata was tracked. Pass metadata_n to forward() or update()."
            )

        keep = store.seen[: store.n] >= min_cells_per_bucket
        if min_metadata_diversity > 1:
            keep &= store.diversity() >= min_metadata_diversity
        if not bool(keep.all()):
            store.filter(keep)
            self._compact_pool()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_cells(self) -> int:
        """Total cells currently stored in the sketch."""
        return 0 if self._store is None else self._store.total_cells()

    @property
    def num_filled_buckets(self) -> int:
        """Number of voxels containing at least one cell."""
        return 0 if self._store is None else self._store.n

    def _voxel_summary(self) -> dict[tuple[int, ...], dict[str, Any]]:
        """
        Inspect the sketch voxel by voxel (for tests and debugging; slow, copies everything to the host).

        Returns:
            A dict from voxel coordinates to ``{"seen": int, "obs_names": [...], "metadata": {...}}``, where
            ``obs_names`` are the voxel's retained cells in order of increasing tag.
        """
        store = self._store
        if store is None or store.n == 0:
            return {}
        coords = store.coords[: store.n].cpu().numpy()
        seen = store.seen[: store.n].cpu().numpy()
        slot_id = store.slot_id[: store.n].cpu().numpy()
        names, _ = self._pool.take(slot_id[slot_id >= 0])
        name_of = dict(zip(slot_id[slot_id >= 0].tolist(), names.tolist()))
        pair_rows, pair_codes = (t.cpu().numpy() for t in store._dedupe_metadata())
        metadata: dict[int, set[int]] = {}
        for row, code in zip(pair_rows.tolist(), pair_codes.tolist()):
            metadata.setdefault(row, set()).add(self._metadata_values[code])
        return {
            tuple(int(c) for c in coords[row]): {
                "seen": int(seen[row]),
                "obs_names": [name_of[i] for i in slot_id[row].tolist() if i >= 0],
                "metadata": metadata.get(row, set()),
            }
            for row in range(store.n)
        }

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def on_train_batch_end(self, trainer: pl.Trainer) -> None:
        super().on_train_batch_end(trainer)
        assert isinstance(trainer.model, pl.LightningModule)
        trainer.model.log("voxel_size_mean", self.voxel_size.mean().item(), prog_bar=True)
        trainer.model.log("voxel_size_max", self.voxel_size.max().item(), prog_bar=True)
        trainer.model.log("coarsen_events", float(self._coarsen_event_count), prog_bar=True)
        trainer.model.log("active_voxels", float(self.num_filled_buckets), prog_bar=True)

    def _final_reservoir_kwargs(self) -> dict[str, Any]:
        return {"max_cells": self.target_n_cells}

    def _storage_state(self) -> dict[str, Any]:
        return {
            "store": None if self._store is None else self._store.state_dict(),
            "pool": self._pool.state_dict(),
            "next_id": self._next_id,
            "max_batch": self._max_batch,
            "coarsen_event_count": self._coarsen_event_count,
            "coarsening_exhausted": self._coarsening_exhausted,
            "metadata_values": list(self._metadata_values),
            "generator_state": self._generator.get_state(),
        }

    def _load_storage_state(self, state: dict[str, Any]) -> None:
        device = self.voxel_offset.device if hasattr(self, "voxel_offset") else torch.device("cpu")
        device = torch.device("cpu") if device.type == "mps" else device
        self._store = None if state["store"] is None else _VoxelStore.from_state_dict(state["store"], device)
        self._pool = _CellPool.from_state_dict(state["pool"])
        self._next_id = state["next_id"]
        self._max_batch = state["max_batch"]
        self._coarsen_event_count = state["coarsen_event_count"]
        self._coarsening_exhausted = state["coarsening_exhausted"]
        self._metadata_values = list(state["metadata_values"])
        self._metadata_code_of = {v: i for i, v in enumerate(self._metadata_values)}
        self._generator.set_state(state["generator_state"])

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Pre-register lazily-shaped buffers with their checkpointed shape before ``load_state_dict`` runs.

        ``voxel_offset`` and ``voxel_size`` are registered lazily by :meth:`_lazy_init` on the
        first minibatch, so a freshly constructed model that hasn't seen a batch yet won't have
        them, causing ``load_state_dict`` to reject them as unexpected keys. This hook runs before Lightning's
        ``load_state_dict`` call (see ``lightning.pytorch.core.saving._load_state``), so it can pre-create same-shaped
        placeholder buffers for each key to load into.
        """
        super().on_load_checkpoint(checkpoint)
        for name in ("voxel_offset", "voxel_size"):
            if hasattr(self, name):
                continue
            for key, tensor in checkpoint["state_dict"].items():
                if key.rsplit(".", 1)[-1] == name:
                    self.register_buffer(name, torch.empty_like(tensor))
                    break

    # ------------------------------------------------------------------
    # Parameter reset
    # ------------------------------------------------------------------

    def _reset_storage(self) -> None:
        self._store = None
        self._pool = _CellPool()
        self._next_id = 0
        self._max_batch = 0
        self._coarsen_event_count = 0
        self._coarsening_exhausted = False
        self._metadata_values = []
        self._metadata_code_of = {}
        # voxel_size and voxel_offset are data-driven (no fixed "initial" value to restore
        # to), so just clear them and let the next _lazy_init recompute both together from
        # the next batch seen.
        if hasattr(self, "voxel_size"):
            delattr(self, "voxel_size")
        if hasattr(self, "voxel_offset"):
            delattr(self, "voxel_offset")
