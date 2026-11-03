# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
The cell collections behind :class:`~cellarium.ml.api.CellariumDataView`: a :class:`ViewCollection` exposes a subset of
the cells of another collection to the datamodule, which therefore needs no knowledge of views, and a
:class:`RamCollection` holds cells in local files that the operating system keeps in memory.
"""

import errno
import os
import shutil
import socket
import uuid
import weakref
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import scipy.sparse as sp
from deltacells.backends import Backend
from deltacells.manifest import OBS_SCHEMA_NAME, Manifest
from deltacells.obs import ObsSchema, ObsStore, resolve_cache_dir
from tqdm import tqdm

from cellarium.ml.data import DeltaCellsBatchCollection, DistributedCollection
from cellarium.ml.data.distributed_deltacells import _obs_columns_needed


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


_DIRECTORY_NAME = "cellarium-ram"  # inside the cache directory: one subdirectory for each RamCollection
_OBS_FINGERPRINT = "ram"  # names the directory of the obs columns inside the directory of a RamCollection


def _gib(n_bytes: float) -> str:
    return f"{n_bytes / 2**30:.2f} GiB"


def _remove_directory(path: str, owner_pid: int) -> None:
    """Delete the directory of a RamCollection, in the process that made it (a forked worker has the collection too)."""
    if os.getpid() == owner_pid:
        shutil.rmtree(path, ignore_errors=True)


def _process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # a process of someone else
        return True
    return True


def _remove_stale_directories(root: Path) -> None:
    """
    Delete the directories in ``root`` (named ``<host>-<pid>-<token>``) of processes on this machine that have ended
    without deleting theirs (killed, for example). Directories of other machines are left alone: the cache can be on a
    disk that machines share, and a process of another machine cannot be looked up here.
    """
    host = socket.gethostname()
    for child in root.iterdir():
        parts = child.name.rsplit("-", 2)
        if len(parts) == 3 and parts[0] == host and parts[1].isdigit() and not _process_is_alive(int(parts[1])):
            shutil.rmtree(child, ignore_errors=True)


def _new_directory(cache_dir: str | os.PathLike | None) -> Path:
    """A new, empty directory for the files of a RamCollection, in the deltacells cache directory."""
    root = resolve_cache_dir(cache_dir) / _DIRECTORY_NAME
    root.mkdir(parents=True, exist_ok=True)
    _remove_stale_directories(root)
    directory = root / f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    directory.mkdir()
    return directory


def _require_free_disk(directory: Path, needed: float) -> None:
    free = shutil.disk_usage(directory).free
    if needed > free:
        raise OSError(
            errno.ENOSPC,
            f"Keeping the cells in memory needs about {_gib(needed)} of free disk space in {directory.parent} "
            f"(they are kept in files there, which the operating system holds in memory), but only {_gib(free)} is "
            "free. Free some space, or choose another directory with cache_dir.",
        )


class _SchemaBackend(Backend):
    """
    The backend of the obs store of a :class:`RamCollection`: all that is read from it is the obs schema, which is held
    here as a string (the obs columns are local files already), so the store can be pickled with it.
    """

    def __init__(self, schema_json: str) -> None:
        self.schema_json = schema_json

    def read(self, name: str) -> bytes:
        if name != OBS_SCHEMA_NAME:
            raise FileNotFoundError(name)
        return self.schema_json.encode()

    def exists(self, name: str) -> bool:
        return name == OBS_SCHEMA_NAME


class RamCollection(DeltaCellsBatchCollection):
    """
    Cells held in local files that the operating system keeps in memory (in its cache of the files): the counts as a
    float32 CSR matrix in files of their own, and the obs in a deltacells :class:`~deltacells.obs.ObsStore`, so the obs
    columns are read, and look, as those of a deltacells dataset do: the categories of categorical columns are global
    strings, and a column is read only when a field needs it. A single shard, so that a datamodule shuffles all of the
    cells together.

    Nothing is held in the memory of the process that made the collection: the files are mapped, so any number of
    dataloader workers read the same memory (a worker process gets a small handle when the collection is pickled, and
    maps the files itself). The files are in a directory of its own in the deltacells cache directory (see
    ``cache_dir`` of :func:`load_into_ram`), which is deleted when the collection that made it is garbage collected or
    the program ends, and is deleted at the next :func:`load_into_ram` if the program was killed. Make one from
    another collection with :func:`load_into_ram`.

    Args:
        directory: The directory with the files that :func:`load_into_ram` wrote: ``data.bin``, ``indices.bin`` and
            ``indptr.bin`` (the CSR matrix) and the files of the obs store.
        n_obs: The number of cells.
        n_vars: The number of genes.
        nnz: The number of nonzero counts.
        obs_store: The obs of the cells.
        var: The ``var`` of the genes.
        var_names: The names of the genes, in column order.
        dropped_obs: The obs columns that could not be kept, by name, with why.
    """

    def __init__(
        self,
        directory: str | os.PathLike,
        n_obs: int,
        n_vars: int,
        nnz: int,
        obs_store: ObsStore,
        var: pd.DataFrame,
        var_names: Any,
        dropped_obs: dict[str, str] | None = None,
    ) -> None:
        self.directory = Path(directory)
        self._n_obs = n_obs
        self._n_vars = n_vars
        self._nnz = nnz
        self._store = obs_store
        self._var = var
        self.var_names = var_names
        self.dropped_obs = dict(dropped_obs or {})
        self.limits = [n_obs]
        self._csr: sp.csr_matrix | None = None
        # only the process that made the files deletes them: a pickled copy (in a worker) is not made here
        self._finalizer: weakref.finalize | None = weakref.finalize(
            self, _remove_directory, str(self.directory), os.getpid()
        )

    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "_csr": None, "_finalizer": None}

    @property
    def n_obs(self) -> int:
        return self._n_obs

    @property
    def n_vars(self) -> int:
        return self._n_vars

    @property
    def var(self) -> pd.DataFrame:
        return self._var

    @property
    def obs_store(self) -> ObsStore:
        return self._store

    @property
    def nbytes(self) -> int:
        """The size of the files that hold the counts and the obs, in bytes."""
        return sum(file.stat().st_size for file in self.directory.rglob("*") if file.is_file())

    def __repr__(self) -> str:
        return f"RamCollection({self.n_obs} cells x {self.n_vars} genes, {_gib(self.nbytes)})"

    def _map(self, name: str, dtype: Any, size: int) -> np.ndarray:
        if size == 0:  # an empty file cannot be mapped
            return np.empty(0, dtype=dtype)
        return np.memmap(self.directory / name, dtype=dtype, mode="r", shape=(size,))

    def _matrix(self) -> sp.csr_matrix:
        """The counts, over the mapped files (made once in each process that reads them)."""
        if self._csr is None:
            data = self._map("data.bin", np.float32, self._nnz)
            indices = self._map("indices.bin", np.int32, self._nnz)
            indptr = self._map("indptr.bin", np.int32, self._n_obs + 1)
            self._csr = sp.csr_matrix((data, indices, indptr), shape=(self._n_obs, self._n_vars))
        return self._csr

    def read_x(self, indices: np.ndarray) -> sp.csr_matrix:
        return self._matrix()[indices]

    def _check_kept(self, columns: Sequence[str]) -> None:
        dropped = [column for column in columns if column in self.dropped_obs]
        if dropped:
            raise KeyError(
                f"The obs columns {dropped} were not kept when the cells were put in memory "
                f"({_dropped_reasons(self.dropped_obs)}). Change their dtype to categorical, numeric, bool or string "
                "in the obs of the data, and call to_ram again."
            )

    def _obs_store(self, key: str) -> ObsStore:
        self._check_kept([key])
        return super()._obs_store(key)

    def take_obs(self, indices: np.ndarray, columns: Sequence[str]) -> pd.DataFrame:
        self._check_kept(columns)
        return super().take_obs(indices, columns)

    def prepare(self, batch_keys: Any) -> None:
        columns = _obs_columns_needed(batch_keys, self.obs_columns_available)
        self._check_kept(columns)
        available = self.obs_columns_available
        missing = [column for column in columns if column not in available]
        if missing:
            raise KeyError(f"Obs columns {missing} are not in the cells in memory; available: {available}")


def _dropped_reasons(dropped: dict[str, str]) -> str:
    return "; ".join(f"{column!r}: {reason}" for column, reason in dropped.items())


def dropped_obs_warning(dropped: dict[str, str]) -> str:
    """The message that tells that obs columns could not be put in memory, and what to do about it."""
    return (
        f"to_ram could not keep {len(dropped)} obs column(s) in memory and left them out "
        f"({_dropped_reasons(dropped)}). A column can be kept only if it is categorical, numeric, bool or string, and "
        "the same in every file of the data. The columns are still in `.obs`, but the cells in memory cannot read them "
        "as fields of a batch. To keep one, change its dtype in the obs of the data (for example "
        "`obs[column] = obs[column].astype(str)` or `.astype('category')`) and call to_ram again."
    )


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


class _Buffer:
    """
    A one-dimensional array that is filled as the counts are read and grows when the estimate of its size was too low.
    With a ``path`` it is a file that is mapped: it grows in place (the file is extended and mapped again, and the
    pages that were written stay in the operating system's cache of the file), so the counts are never held twice.
    Without one it is an array in the memory of the process, which grows by copying (the old and the new array exist
    at once for a moment).
    """

    def __init__(self, dtype: Any, capacity: int, path: Path | None) -> None:
        self.dtype = np.dtype(dtype)
        self.path = path
        self.array: np.ndarray | None = np.empty(capacity, self.dtype) if path is None else self._map(capacity)

    def _map(self, capacity: int) -> np.ndarray:
        assert self.path is not None
        with open(self.path, "r+b" if self.path.exists() else "wb") as file:
            file.truncate(capacity * self.dtype.itemsize)
        return np.memmap(self.path, dtype=self.dtype, mode="r+", shape=(capacity,))

    def resize(self, capacity: int, used: int) -> None:
        """Make room for ``capacity`` elements, keeping the first ``used``."""
        assert self.array is not None
        if self.path is None:
            new = np.empty(capacity, self.dtype)
            new[:used] = self.array[:used]
            self.array = new
        else:
            self.array = None  # unmaps the old pages; what was written to them stays in the file
            self.array = self._map(capacity)

    def finish(self, used: int) -> None:
        """Cut the array down to the ``used`` elements that were written (a file is closed: map it to read it)."""
        assert self.array is not None
        if self.path is None:
            self.array.resize(used, refcheck=False)
        else:
            self.array = None
            with open(self.path, "r+b") as file:
                file.truncate(used * self.dtype.itemsize)


def shard_blocks(collection: DistributedCollection, chunk_size: int) -> list[tuple[int, int]]:
    """
    The runs of consecutive cells of ``collection`` to read it in order, as ``(start, stop)``: each lies in one shard
    (a file, or a tile of a deltacells dataset; for a view, the cells of the view that lie in one shard of the data it
    is taken from), and has at most ``chunk_size`` cells. Reading a run touches one shard, whichever cells of the data
    the collection holds, so a pass over them reads and decodes each shard once.
    """
    n = len(collection)
    blocks = []
    lower = 0
    for upper in collection.limits:
        upper = min(upper, n)
        for start in range(lower, upper, chunk_size):
            blocks.append((start, min(start + chunk_size, upper)))
        lower = max(lower, upper)
    if lower < n:  # a collection whose limits stop short of its cells
        blocks.extend((start, min(start + chunk_size, n)) for start in range(lower, n, chunk_size))
    return blocks


def read_blocks(
    collection: DistributedCollection,
    blocks: Sequence[tuple[int, int]],
    lookahead: int = 3,
    progress: bool = True,
    desc: str = "Reading cells",
) -> Iterator[tuple[int, int, Any]]:
    """
    Read the ``blocks`` of cells of ``collection`` (see :func:`shard_blocks`) in order: for each, its ``start``,
    ``stop`` and the batch of its cells. The next ``lookahead`` blocks are announced to the collection before a block
    is read (see :meth:`~cellarium.ml.data.DistributedCollection.prefetch`), so that a collection that can fetch in the
    background reads the next shards while this one is used.
    """
    announced = -1
    iterator: Iterable[int] = range(len(blocks))
    if progress:
        iterator = tqdm(iterator, desc=desc, unit="block", total=len(blocks))
    for i in iterator:
        while announced < min(i + lookahead, len(blocks) - 1):
            announced += 1
            collection.prefetch(np.arange(*blocks[announced]))
        start, stop = blocks[i]
        yield start, stop, collection.read(np.arange(start, stop))


def _estimate(
    collection: DistributedCollection, blocks: Sequence[tuple[int, int]], n_samples: int, sample_size: int
) -> tuple[float, float]:
    """The mean number of nonzeros and the mean bytes of obs of a cell, from the first cells of blocks that are spread
    over the collection: ``n_samples`` blocks, or (if they are smaller than ``sample_size``, as the cells of a view of
    scattered cells in each shard are) as many as it takes for about ``n_samples * sample_size`` cells, up to 32."""
    sizes = np.array([min(sample_size, stop - start) for start, stop in blocks])
    wanted = int(np.ceil(n_samples * sample_size / sizes.mean()))
    n_blocks = min(len(blocks), max(n_samples, wanted), max(n_samples, 32))
    picks = np.unique(np.linspace(0, len(blocks) - 1, n_blocks).round().astype(np.int64))
    nnz = obs_bytes = n_cells = 0
    for i in picks:
        start, _ = blocks[i]
        sample = collection.read(np.arange(start, start + sizes[i]))
        nnz += _to_csr32(sample.X).nnz
        obs_bytes += int(_obs_frame(sample).memory_usage(deep=True).sum())
        n_cells += int(sizes[i])
    return nnz / n_cells, obs_bytes / n_cells


def _load_counts(
    collection: DistributedCollection,
    max_bytes: int,
    chunk_size: int,
    n_samples: int,
    sample_size: int,
    progress: bool,
    directory: Path | None,
) -> tuple[_Buffer, _Buffer, np.ndarray, list[pd.DataFrame]]:
    """
    Read the counts of all cells of ``collection`` (one pass, in order), with the checks of memory that
    :func:`load_into_ram` describes: into the files ``data.bin`` and ``indices.bin`` of ``directory`` or, without one,
    into arrays in memory. Returns the buffers of the values and of the column indices (cut to what was read: with a
    directory the files are closed), the row pointers (int32), and, with a directory, the obs of each block that was
    read.
    """
    n = len(collection)

    def too_big(needed: float, estimated: bool) -> MemoryError:
        return MemoryError(
            f"Loading the {n} cells into memory {'would need about' if estimated else 'needs more than'} "
            f"{_gib(needed)}, but at most {_gib(max_bytes)} may be used (by default half of the available "
            "memory). Make the view smaller, or raise the limit with max_gb."
        )

    blocks = shard_blocks(collection, chunk_size)
    nnz_per_cell, obs_per_cell = _estimate(collection, blocks, n_samples, sample_size)
    # a float32 value and an int32 column index per nonzero, an int32 row pointer per cell
    per_cell = nnz_per_cell * 8 + 4 + obs_per_cell
    if n * per_cell > max_bytes:
        raise too_big(n * per_cell, estimated=True)
    if n * nnz_per_cell > 1.1 * (2**31 - 1):
        raise MemoryError(
            f"The {n} cells have about {n * nnz_per_cell:.2e} nonzero counts, more than the 2**31 - 1 that a matrix "
            "with 32-bit indices can hold. Make the view smaller."
        )

    if directory is not None:
        _require_free_disk(directory, n * per_cell)

    capacity = int(n * nnz_per_cell * 1.1) + chunk_size
    data = _Buffer(np.float32, capacity, None if directory is None else directory / "data.bin")
    indices = _Buffer(np.int32, capacity, None if directory is None else directory / "indices.bin")
    indptr = np.zeros(n + 1, dtype=np.int64)
    nnz, obs_bytes, parts = 0, 0, []
    for start, stop, batch in read_blocks(collection, blocks, progress=progress, desc="Loading cells into memory"):
        x = _to_csr32(batch.X)
        obs = _obs_frame(batch)
        if nnz + x.nnz > capacity:
            new_capacity = max(nnz + x.nnz, int(capacity * 1.25))
            if new_capacity * 8 + indptr.nbytes + obs_bytes > max_bytes:
                raise too_big(new_capacity * 8 + indptr.nbytes + obs_bytes, estimated=False)
            if directory is not None:
                _require_free_disk(directory, (new_capacity - capacity) * 8)
            capacity = new_capacity
            data.resize(capacity, nnz)
            indices.resize(capacity, nnz)
        assert data.array is not None and indices.array is not None
        data.array[nnz : nnz + x.nnz] = x.data
        indices.array[nnz : nnz + x.nnz] = x.indices
        indptr[start + 1 : stop + 1] = nnz + x.indptr[1:]
        nnz += x.nnz
        obs_bytes += int(obs.memory_usage(deep=True).sum())
        if directory is not None:
            parts.append(obs)
        if nnz * 8 + indptr.nbytes + obs_bytes > max_bytes:
            raise too_big(nnz * 8 + indptr.nbytes + obs_bytes, estimated=False)
    if nnz >= 2**31:
        raise too_big(nnz * 8, estimated=False)  # int32 column indices and row pointers cannot hold more
    data.finish(nnz)
    indices.finish(nnz)
    return data, indices, indptr.astype(np.int32), parts


def load_counts(
    collection: DistributedCollection,
    max_bytes: int,
    chunk_size: int = 10_000,
    n_samples: int = 5,
    sample_size: int = 1_000,
    progress: bool = True,
) -> sp.csr_matrix:
    """
    Read the counts of all cells of ``collection`` into a float32 CSR matrix in the memory of this process, as
    :func:`load_into_ram` reads them into files (see it for the way the cells are read and the memory is checked).

    Raises:
        MemoryError: If the counts do not fit in ``max_bytes``.
    """
    data, indices, indptr, _ = _load_counts(collection, max_bytes, chunk_size, n_samples, sample_size, progress, None)
    assert data.array is not None and indices.array is not None
    return sp.csr_matrix((data.array, indices.array, indptr), shape=(len(collection), collection.n_vars))


def _write_obs(
    parts: list[pd.DataFrame], directory: Path, n_obs: int, n_vars: int, nnz: int
) -> tuple[ObsStore, dict[str, str]]:
    """
    Encode the obs of the blocks of cells that were read (``parts``, each indexed by the cell names) as a deltacells
    dataset stores obs, and write them to ``directory`` the way the obs cache of a deltacells dataset is laid out (one
    file for each column), so that an :class:`~deltacells.obs.ObsStore` reads them. The cells are one "tile", so that a
    batch of cells from all over them takes one lookup for each column. Returns the store, and the columns that cannot
    be stored (by name, with why), which are left out.
    """
    frames = [part.assign(obs_names=np.asarray(part.index, dtype=str)).reset_index(drop=True) for part in parts]
    parts.clear()
    specs, dropped = [], {}
    for name in frames[0].columns:
        try:
            specs.extend(ObsSchema.infer(frames, columns=[name]).columns)
        except ValueError as error:
            dropped[str(name)] = (
                f"dtype {frames[0][name].dtype}" if "unsupported dtype" in str(error) else str(error).rstrip(".")
            )
    schema = ObsSchema(specs)
    table = pa.concat_tables([schema.encode(frame[schema.names]) for frame in frames])
    del frames
    manifest = Manifest(
        n_cells=n_obs,
        n_genes=n_vars,
        tile_size=n_obs,
        tile_cells=[n_obs],
        tile_nnz=[nnz],
        tile_bytes=[0],
        n_chunks=1,
        zstd_level=0,
        has_obs=True,
        obs_tiles_per_file=1,
        obs_fingerprint=_OBS_FINGERPRINT,
        obs_shard_bytes=[0],
        obs_shard_sha256=[""],
    )
    store = ObsStore(_SchemaBackend(schema.to_json()), manifest, cache_dir=directory)
    store.cache_dir.mkdir(parents=True)
    for spec in schema.columns:
        column = table.column(spec.name).combine_chunks()
        with pa.OSFile(str(store._path(spec.name)), "wb") as sink:
            writer = pa.ipc.new_file(sink, pa.schema([pa.field(spec.name, column.type)]))
            writer.write_batch(pa.record_batch([column], names=[spec.name]))
            writer.close()
    return store, dropped


def load_into_ram(
    collection: DistributedCollection,
    max_bytes: int,
    chunk_size: int = 10_000,
    n_samples: int = 5,
    sample_size: int = 1_000,
    progress: bool = True,
    cache_dir: str | os.PathLike | None = None,
) -> RamCollection:
    """
    Read all cells of ``collection`` (one pass, in order) into a :class:`RamCollection`.

    The cells are read one shard at a time (see :func:`shard_blocks`; a shard with more than ``chunk_size`` cells is
    read in pieces), so that each shard is read and decoded once, with the next few announced to the collection ahead
    of time, even if the cells are scattered over many shards (as those of a view can be) and a read of ``chunk_size``
    cells in a row would touch most of them.

    The memory needed is estimated first, from ``n_samples`` blocks of up to ``sample_size`` cells spread over the
    collection, and nothing is loaded if it is more than ``max_bytes`` (or if the counts have clearly more nonzeros than
    the 32-bit indices of the matrix can hold), or than the free disk space of the directory that the cells are written
    to. While loading, the memory in use is checked after every block, so a bad estimate cannot run the machine out of
    memory either. The counts are written straight into files, which are sized from the estimate and extended in place
    if it was too low, so they are never held twice.

    The obs columns that cannot be stored (see :func:`dropped_obs_warning`) are left out, and named in the
    ``dropped_obs`` of the result.

    Args:
        collection: The cells to read.
        max_bytes: The most memory that they may take.
        chunk_size: The most cells to read at a time.
        n_samples: The number of blocks that the memory needed is estimated from.
        sample_size: The most cells in each of those blocks.
        progress: Whether to show a progress bar.
        cache_dir: The deltacells cache directory (see :func:`deltacells.obs.resolve_cache_dir`) that the directory with
            the files of the cells is made in: ``$DELTACELLS_CACHE`` or ``~/.cache/deltacells`` by default.

    Raises:
        MemoryError: If the cells do not fit in ``max_bytes``.
        OSError: If there is not enough free disk space.
    """
    directory = _new_directory(cache_dir)
    try:
        _, _, indptr, parts = _load_counts(
            collection, max_bytes, chunk_size, n_samples, sample_size, progress, directory
        )
        nnz = int(indptr[-1])
        indptr.tofile(directory / "indptr.bin")
        store, dropped = _write_obs(parts, directory, len(collection), collection.n_vars, nnz)
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return RamCollection(
        directory, len(collection), collection.n_vars, nnz, store, collection.var, collection.var_names, dropped
    )
