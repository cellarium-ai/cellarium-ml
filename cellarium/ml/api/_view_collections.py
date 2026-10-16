# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
The cell collections behind :class:`~cellarium.ml.api.CellariumDataView`: a :class:`ViewCollection` exposes a subset of
the cells of another collection to the datamodule, which therefore needs no knowledge of views, and a
:class:`RamCollection` holds cells in memory.
"""

from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm

from cellarium.ml.data import DistributedCollection


class ViewCollection(DistributedCollection):
    """
    The cells ``positions`` (sorted, unique) of the collection ``source``, as a collection of their own.

    Cell ``i`` of the view is cell ``positions[i]`` of the source. A shard of the view is the view's cells that lie in
    one shard of the source (shards without a view cell are dropped), so a datamodule that shuffles shards and then the
    cells within them reads each shard of the source once per epoch, as it does for the source itself. Nothing is
    copied: reads go to the source, and so do the properties that do not depend on the cells (``var``, the categories
    of obs columns, ...).

    A view of a view is flattened: the source is always a collection that is not itself a :class:`ViewCollection`.

    Args:
        source: The collection to take the cells from.
        positions: Positions in ``source`` of the cells of the view; strictly increasing.
    """

    source: DistributedCollection
    positions: np.ndarray

    def __init__(self, source: DistributedCollection, positions: np.ndarray) -> None:
        positions = np.asarray(positions, dtype=np.int64)
        if positions.ndim != 1:
            raise ValueError(f"positions must be one-dimensional, got shape {positions.shape}")
        if len(positions) == 0:
            raise ValueError("A view must contain at least one cell.")
        if np.any(positions[1:] <= positions[:-1]):
            raise ValueError("positions must be strictly increasing (sorted and unique).")
        if positions[0] < 0 or positions[-1] >= len(source):
            raise IndexError(f"positions must lie in [0, {len(source)}), got {positions[0]} to {positions[-1]}")
        if isinstance(source, ViewCollection):
            positions = source.positions[positions]
            source = source.source
        self.source = source
        self.positions = positions
        self.var_names = source.var_names
        self.supports_prefetch = source.supports_prefetch
        # cells of the view before the end of each shard of the source, without the shards that hold no view cell
        cumulative = np.searchsorted(positions, np.asarray(source.limits), side="left")
        previous = np.concatenate(([0], cumulative[:-1]))
        self.limits = [int(limit) for limit in cumulative[cumulative > previous]]

    def __repr__(self) -> str:
        return f"ViewCollection({self.n_obs} of the {len(self.source)} cells of {self.source!r})"

    @property
    def n_obs(self) -> int:
        return len(self.positions)

    @property
    def n_vars(self) -> int:
        return self.source.n_vars

    @property
    def var(self) -> pd.DataFrame:
        return self.source.var

    @property
    def reference_adata(self) -> Any:
        return self.source.reference_adata

    def _source_positions(self, index: Any) -> np.ndarray:
        """The positions in the source of the cells of the view at ``index`` (an int, a slice or a sequence)."""
        n = self.n_obs
        if isinstance(index, slice):
            return self.positions[np.arange(*index.indices(n), dtype=np.int64)]
        if isinstance(index, (int, np.integer)):
            i = int(index) + n if index < 0 else int(index)
            if not 0 <= i < n:
                raise IndexError(f"Cell {index} is out of range for a view of {n} cells")
            return self.positions[[i]]
        idx = np.asarray(index)
        if idx.dtype == np.bool_:
            if idx.shape != (n,):
                raise IndexError(f"A boolean index must have shape ({n},), got {idx.shape}")
            return self.positions[idx]
        idx = idx.astype(np.int64, copy=False).ravel()
        if len(idx) and (idx.min() < -n or idx.max() >= n):
            raise IndexError(f"Cell indices must lie in [-{n}, {n}), got {idx.min()} to {idx.max()}")
        return self.positions[np.where(idx < 0, idx + n, idx)]

    def __getitem__(self, index: Any) -> Any:
        # the cells of a batch can lie in more shards than the source's cache holds
        return self.source.read(self._source_positions(index))

    def prefetch(self, indices: np.ndarray) -> None:
        self.source.prefetch(self._source_positions(indices))

    def prepare(self, batch_keys: Any) -> None:
        self.source.prepare(batch_keys)

    def reset_cache(self) -> None:
        self.source.reset_cache()

    @property
    def cache_miss_count(self) -> int:
        return getattr(self.source, "cache_miss_count", 0)

    def obs_categories(self, key: str) -> np.ndarray:
        return self.source.obs_categories(key)

    def obs_key_nunique(self, key: str) -> int:
        return self.source.obs_key_nunique(key)


def unwrap_collection(dadc: Any) -> Any:
    """The collection that actually holds the cells: ``dadc`` itself unless it is a :class:`ViewCollection`."""
    while isinstance(dadc, ViewCollection):
        dadc = dadc.source
    return dadc


class RamBatch:
    """
    A batch of cells of a :class:`RamCollection`, behaving like an :class:`~anndata.AnnData` for the attributes that
    :class:`~cellarium.ml.utilities.data.AnnDataField` reads: ``X`` (a CSR matrix), ``obs``, ``obs_names``, ``var`` and
    ``var_names``.
    """

    def __init__(self, X: sp.csr_matrix, obs: pd.DataFrame, var: pd.DataFrame, var_names: Any) -> None:
        self.X = X
        self.obs = obs
        self.var = var
        self.var_names = var_names

    @property
    def obs_names(self) -> pd.Index:
        return self.obs.index

    @property
    def n_obs(self) -> int:
        return self.X.shape[0]

    @property
    def n_vars(self) -> int:
        return self.X.shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        return self.X.shape

    def __len__(self) -> int:
        return self.n_obs


class RamCollection(DistributedCollection):
    """
    Cells held in memory: a float32 CSR matrix of the counts and a frame with all of their ``obs`` columns. A single
    shard, so that a datamodule shuffles all of the cells together. Since the cells live in this process, no
    dataloader workers can be used (``max_num_workers = 0``). Make one from another collection with
    :func:`load_into_ram`.

    Args:
        x: The counts, one row per cell.
        obs: The ``obs`` of the cells, indexed by their names.
        var: The ``var`` of the genes.
        var_names: The names of the genes, in column order.
    """

    max_num_workers = 0

    def __init__(self, x: sp.csr_matrix, obs: pd.DataFrame, var: pd.DataFrame, var_names: Any) -> None:
        if x.shape[0] != len(obs):
            raise ValueError(f"x has {x.shape[0]} cells but obs has {len(obs)}.")
        self.x = x
        self.obs = obs
        self._var = var
        self.var_names = var_names
        self.limits = [x.shape[0]]

    @property
    def n_obs(self) -> int:
        return self.x.shape[0]

    @property
    def n_vars(self) -> int:
        return self.x.shape[1]

    @property
    def var(self) -> pd.DataFrame:
        return self._var

    @property
    def nbytes(self) -> int:
        """The memory the counts and the obs take, in bytes."""
        return self.x.data.nbytes + self.x.indices.nbytes + self.x.indptr.nbytes + int(self.obs.memory_usage().sum())

    def __repr__(self) -> str:
        return f"RamCollection({self.n_obs} cells x {self.n_vars} genes, {self.nbytes / 2**30:.2f} GiB)"

    def __getitem__(self, index: Any) -> RamBatch:
        if isinstance(index, (int, np.integer)):
            index = [index]
        elif not isinstance(index, slice):
            index = np.asarray(index)
            if index.dtype != np.bool_:
                index = index.astype(np.int64, copy=False).ravel()
        return RamBatch(self.x[index], self.obs.iloc[index], self._var, self.var_names)

    def obs_categories(self, key: str) -> np.ndarray:
        return np.asarray(self.obs[key].cat.categories)

    def obs_key_nunique(self, key: str) -> int:
        column = self.obs[key]
        return len(column.cat.categories) if isinstance(column.dtype, pd.CategoricalDtype) else int(column.nunique())


def _cgroup_available_bytes() -> int | None:
    """The memory this process may still use according to its cgroup (a container limit), or ``None`` if unlimited."""
    for limit_file, usage_file in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        try:
            with open(limit_file) as f:
                limit = f.read().strip()
            with open(usage_file) as f:
                usage = int(f.read().strip())
        except (OSError, ValueError):
            continue
        if limit == "max" or int(limit) >= 2**60:  # no limit
            return None
        return max(int(limit) - usage, 0)
    return None


def available_memory_bytes() -> int:
    """
    The memory available to this process: what the system reports as available, or less if a container limit
    (cgroup) applies.

    Raises:
        ValueError: If it cannot be determined on this system.
    """
    candidates = []
    try:
        import psutil

        candidates.append(int(psutil.virtual_memory().available))
    except ImportError:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        candidates.append(int(line.split()[1]) * 1024)
        except OSError:
            pass
    cgroup = _cgroup_available_bytes()
    if cgroup is not None:
        candidates.append(cgroup)
    if not candidates:
        raise ValueError("Cannot determine the available memory on this system; pass max_gb.")
    return min(candidates)


def _obs_frame(batch: Any) -> pd.DataFrame:
    """All the ``obs`` columns of a batch of cells, indexed by the cell names."""
    obs = batch.obs
    columns = [c for c in obs.columns if c != "obs_names"]
    frame = obs[columns].copy() if columns else pd.DataFrame(index=pd.RangeIndex(batch.n_obs))
    frame.index = pd.Index(batch.obs_names, name=None)
    return frame


def _to_csr32(x: Any) -> sp.csr_matrix:
    return sp.csr_matrix(x, dtype=np.float32)


def _grown(array: np.ndarray, capacity: int, used: int) -> np.ndarray:
    """A new array of ``capacity`` elements that starts with the first ``used`` elements of ``array``."""
    new = np.empty(capacity, dtype=array.dtype)
    new[:used] = array[:used]
    return new


def _gib(n_bytes: float) -> str:
    return f"{n_bytes / 2**30:.2f} GiB"


def load_into_ram(
    collection: DistributedCollection,
    max_bytes: int,
    chunk_size: int = 10_000,
    n_samples: int = 5,
    sample_size: int = 1_000,
    progress: bool = True,
) -> RamCollection:
    """
    Read all cells of ``collection`` (one pass, in order) into a :class:`RamCollection`.

    The memory needed is estimated first, from ``n_samples`` blocks of ``sample_size`` cells spread over the
    collection, and nothing is loaded if it is more than ``max_bytes``. While loading, the memory in use is checked
    after every chunk of ``chunk_size`` cells, so a bad estimate cannot run the machine out of memory either. (The
    counts are written into arrays sized from the estimate, which are grown if it was too low; in that case
    the old and new arrays coexist for a moment.)

    Raises:
        MemoryError: If the cells do not fit in ``max_bytes``.
    """
    n = len(collection)

    def too_big(needed: float, estimated: bool) -> MemoryError:
        return MemoryError(
            f"Loading the {n} cells into memory {'would need about' if estimated else 'needs more than'} "
            f"{_gib(needed)}, but at most {_gib(max_bytes)} may be used (by default half of the available "
            "memory). Make the view smaller, or raise the limit with max_gb."
        )

    # estimate from blocks of cells spread over the collection
    block = min(sample_size, n)
    starts = np.unique(np.linspace(0, n - block, min(n_samples, max(n // block, 1))).astype(np.int64))
    nnz_per_cell = obs_per_cell = 0.0
    for start in starts:
        sample = collection.read(np.arange(start, start + block))
        nnz_per_cell += _to_csr32(sample.X).nnz / block / len(starts)
        obs_per_cell += int(_obs_frame(sample).memory_usage(deep=True).sum()) / block / len(starts)
    # a float32 value and an int32 column index per nonzero, an int32 row pointer per cell
    per_cell = nnz_per_cell * 8 + 4 + obs_per_cell
    if n * per_cell > max_bytes:
        raise too_big(n * per_cell, estimated=True)

    capacity = int(n * nnz_per_cell * 1.1) + chunk_size
    data = np.empty(capacity, dtype=np.float32)
    indices = np.empty(capacity, dtype=np.int32)
    indptr = np.zeros(n + 1, dtype=np.int64)
    nnz, obs_bytes, parts = 0, 0, []
    starts_of_chunks = range(0, n, chunk_size)
    if progress:
        starts_of_chunks = tqdm(starts_of_chunks, desc="Loading cells into memory", unit="chunk")  # type: ignore[assignment]
    for start in starts_of_chunks:
        stop = min(start + chunk_size, n)
        batch = collection.read(np.arange(start, stop))
        x = _to_csr32(batch.X)
        obs = _obs_frame(batch)
        if nnz + x.nnz > capacity:
            capacity = max(nnz + x.nnz, int(capacity * 1.25))
            if capacity * 8 + indptr.nbytes + obs_bytes > max_bytes:
                raise too_big(capacity * 8 + indptr.nbytes + obs_bytes, estimated=False)
            data = _grown(data, capacity, nnz)
            indices = _grown(indices, capacity, nnz)
        data[nnz : nnz + x.nnz] = x.data
        indices[nnz : nnz + x.nnz] = x.indices
        indptr[start + 1 : stop + 1] = nnz + x.indptr[1:]
        nnz += x.nnz
        obs_bytes += int(obs.memory_usage(deep=True).sum())
        parts.append(obs)
        if nnz * 8 + indptr.nbytes + obs_bytes > max_bytes:
            raise too_big(nnz * 8 + indptr.nbytes + obs_bytes, estimated=False)
    data.resize(nnz, refcheck=False)
    indices.resize(nnz, refcheck=False)
    if nnz >= 2**31:
        raise too_big(nnz * 8, estimated=False)  # int32 column indices and row pointers cannot hold more
    x_all = sp.csr_matrix((data, indices, indptr.astype(np.int32)), shape=(n, collection.n_vars))
    return RamCollection(x_all, pd.concat(parts), collection.var, collection.var_names)
