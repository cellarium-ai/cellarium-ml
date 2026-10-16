# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""A dataset of tiles, with tile prefetching, a decoded-tile cache with buffer reuse, and batch gathering across tiles."""

from __future__ import annotations

import io
import threading
import time
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import numpy as np

from deltacells.backends import Backend, open_backend
from deltacells.manifest import GENE_ORDER_NAME, MANIFEST_NAME, VAR_NAMES_NAME, Manifest
from deltacells.reader import DecodedTile, SparseBatch, Tile, _check_out, as_values_dtype, gather_into


class DeltaCellsDataset:
    """Random access to the cells of a deltacells dataset, designed to sit behind a ``DataLoader`` worker.

    The access pattern it is built for is the one h5ad shards already get: batches whose cells come from one or two tiles at a
    time. ``get_batch`` takes global cell indices (any order, repeats allowed, any number of tiles) and returns a CSR
    :class:`SparseBatch` in that order. Whole tiles are fetched (one object read each) and decoded; the compressed tiles and the
    decoded tiles are cached (LRU), and ``prefetch`` lets the caller announce which tiles it will need next so that their
    fetching and parsing overlaps with decoding and with the training step.

    Args:
        uri: Dataset location: a local path, ``gs://bucket/prefix`` or a :class:`~deltacells.backends.Backend`.
        max_cached_tiles: Decoded tiles kept in memory (each ~ 8 bytes per nonzero; ~0.4 GB for 10k deep cells).
        max_prefetch_tiles: Compressed tiles kept (or in flight) in the fetch cache.
        io_threads: Concurrent tile fetches.
        decode_threads: Chunks of one tile decoded in parallel (the C core releases the GIL). Use 1 when each DataLoader worker
            already gets about one core.
        verify: Check each tile's CRC32 when it is parsed (done on the fetch threads).
        cache_dir: Where localized obs columns are cached (default: ``$DELTACELLS_CACHE`` or ``~/.cache/deltacells``).
        obs_columns: Obs columns to make local right away (see :attr:`obs`); more can be added any time.

    The object is picklable (caches, threads and buffers are rebuilt lazily), so it can be passed to DataLoader workers. It is
    not safe to call ``get_batch`` concurrently from several threads of one process (calls are serialized by a lock).
    """

    def __init__(
        self,
        uri: str | Backend,
        *,
        max_cached_tiles: int = 2,
        max_prefetch_tiles: int = 8,
        io_threads: int = 4,
        decode_threads: int = 1,
        verify: bool = False,
        cache_dir: str | None = None,
        obs_columns: Sequence[str] | None = None,
    ) -> None:
        if max_cached_tiles < 1 or max_prefetch_tiles < 1 or io_threads < 1 or decode_threads < 1:
            raise ValueError("max_cached_tiles, max_prefetch_tiles, io_threads and decode_threads must be at least 1")
        self.backend = open_backend(uri)
        self.max_cached_tiles, self.max_prefetch_tiles = max_cached_tiles, max_prefetch_tiles
        self.io_threads, self.decode_threads, self.verify = io_threads, decode_threads, verify
        self.cache_dir = cache_dir
        self.manifest = Manifest.from_json(bytes(self.backend.read(MANIFEST_NAME)))
        self._init_runtime()
        if obs_columns:
            if self.obs is None:
                raise ValueError("obs_columns were given but the dataset has no obs")
            self.obs.localize(obs_columns)

    # ------------------------------------------------------------------ pickling / lifecycle

    def _init_runtime(self) -> None:
        self._limits = self.manifest.limits
        self._starts = np.concatenate([[0], self._limits[:-1]]).astype(np.int64)
        self._lock = threading.RLock()  # guards the caches; never held by the fetch threads
        self._stats_lock = threading.Lock()
        self._tiles: OrderedDict[int, Future] = OrderedDict()  # compressed + parsed tiles (futures)
        self._decoded: OrderedDict[int, tuple[DecodedTile, tuple[np.ndarray, np.ndarray]]] = OrderedDict()
        self._free_buffers: list[tuple[np.ndarray, np.ndarray]] = []
        self._scratch = np.empty(0, dtype=np.uint8)
        self._io_pool: ThreadPoolExecutor | None = None
        self._decode_pool: ThreadPoolExecutor | None = None
        self._var_names: list[str] | None = None
        self._gene_order: np.ndarray | None = None
        self._obs = None
        self._var = None
        self.stats = {"tiles_fetched": 0, "fetch_seconds": 0.0, "stall_seconds": 0.0, "tiles_decoded": 0, "decode_seconds": 0.0, "tile_cache_hits": 0, "decoded_cache_hits": 0}  # fmt: skip

    def __getstate__(self) -> dict[str, Any]:
        return {
            "backend": self.backend, "manifest": self.manifest, "max_cached_tiles": self.max_cached_tiles,
            "max_prefetch_tiles": self.max_prefetch_tiles, "io_threads": self.io_threads,
            "decode_threads": self.decode_threads, "verify": self.verify, "cache_dir": self.cache_dir,
        }  # fmt: skip

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._init_runtime()

    def close(self) -> None:
        """Stop the helper threads and drop all caches."""
        with self._lock:
            for pool in (self._io_pool, self._decode_pool):
                if pool is not None:
                    pool.shutdown(wait=True, cancel_futures=True)
            self._io_pool = self._decode_pool = None
            if self._obs is not None:
                self._obs.close()
                self._obs = None
            self._tiles.clear()
            self._decoded.clear()
            self._free_buffers.clear()

    def __enter__(self) -> DeltaCellsDataset:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ metadata

    @property
    def n_cells(self) -> int:
        return self.manifest.n_cells

    @property
    def n_genes(self) -> int:
        return self.manifest.n_genes

    @property
    def n_tiles(self) -> int:
        return self.manifest.n_tiles

    @property
    def tile_size(self) -> int:
        return self.manifest.tile_size

    @property
    def limits(self) -> np.ndarray:
        """Cumulative cell counts per tile (compare ``DistributedAnnDataCollection.limits``)."""
        return self._limits

    def __len__(self) -> int:
        return self.n_cells

    def tile_bounds(self, i: int) -> tuple[int, int]:
        """Global ``(start, stop)`` cell indices of tile ``i``."""
        return int(self._starts[i]), int(self._limits[i])

    def tile_of(self, indices: Sequence[int] | np.ndarray) -> np.ndarray:
        """Tile index of each global cell index."""
        return np.searchsorted(self._limits, np.asarray(indices, dtype=np.int64), side="right")

    @property
    def var_names(self) -> list[str] | None:
        """Gene names in column order (``None`` if the dataset was written without them)."""
        if self._var_names is None and self.manifest.has_var_names:
            self._var_names = bytes(self.backend.read(VAR_NAMES_NAME)).decode().splitlines()
        return self._var_names

    @property
    def gene_order(self) -> np.ndarray | None:
        """``gene_order[j]`` is the input-file column that became column ``j`` (``None`` if genes were not reordered)."""
        if self._gene_order is None and self.manifest.has_gene_order:
            self._gene_order = np.load(io.BytesIO(bytes(self.backend.read(GENE_ORDER_NAME))))
        return self._gene_order

    @property
    def obs(self):
        """The dataset's per-cell metadata as a :class:`deltacells.obs.ObsStore` (``None`` if it has none; needs pyarrow and pandas).

        Obs is never fetched with the tiles. ``obs.localize(columns)`` copies just those columns into a node-local cache (once;
        more columns can be added later), after which ``obs.take(indices, columns)`` is a memory-mapped lookup.
        """
        if not self.manifest.has_obs:
            return None
        if self._obs is None:
            from deltacells.obs import ObsStore

            self._obs = ObsStore(self.backend, self.manifest, cache_dir=self.cache_dir)
        return self._obs

    @property
    def var(self):
        """The gene table as a pandas DataFrame in column order (``None`` if the dataset was written without one)."""
        if self._var is None and self.manifest.has_var_table:
            from deltacells.obs import read_var_table

            self._var = read_var_table(self.backend)
        return self._var

    # ------------------------------------------------------------------ fetching and prefetching

    def _io(self) -> ThreadPoolExecutor:
        if self._io_pool is None:
            self._io_pool = ThreadPoolExecutor(self.io_threads, thread_name_prefix="deltacells-io")
        return self._io_pool

    def _fetch_tile(self, i: int) -> Tile:
        t0 = time.perf_counter()
        tile = Tile(self.backend.read(Manifest.tile_name(i)), verify=self.verify)
        if tile.n_cells != self.manifest.tile_cells[i] or tile.n_genes != self.n_genes:
            raise ValueError(f"tile {i} does not match the manifest")
        with self._stats_lock:
            self.stats["tiles_fetched"] += 1
            self.stats["fetch_seconds"] += time.perf_counter() - t0
        return tile

    def _evict_tiles(self, pinned: set[int]) -> bool:
        """Make room in the compressed-tile cache; returns False if everything is pinned or still in flight."""
        while len(self._tiles) >= self.max_prefetch_tiles:
            victim = next((k for k, f in self._tiles.items() if k not in pinned and f.done()), None)
            if victim is None:
                return False
            del self._tiles[victim]
        return True

    def prefetch(self, tile_ids: Iterable[int]) -> int:
        """Start fetching tiles in the background (non-blocking, idempotent). Returns the number of fetches started.

        Tiles that do not fit in ``max_prefetch_tiles`` (because the cache holds pinned or in-flight tiles) are skipped.
        """
        ids = list(dict.fromkeys(int(t) for t in tile_ids))
        started = 0
        with self._lock:
            pinned = set(ids)
            for i in ids:
                if not 0 <= i < self.n_tiles:
                    raise IndexError(f"tile {i} out of range")
                if i in self._tiles:
                    continue
                if not self._evict_tiles(pinned):
                    break
                self._tiles[i] = self._io().submit(self._fetch_tile, i)
                started += 1
        return started

    def prefetch_cells(self, indices: Sequence[int] | np.ndarray) -> int:
        """:meth:`prefetch` for the tiles containing the given global cell indices."""
        idx = np.asarray(indices, dtype=np.int64)
        if len(idx) == 0:
            return 0
        if idx.min() < 0 or idx.max() >= self.n_cells:
            raise IndexError("cell index out of range")
        return self.prefetch(self.tile_of(idx).tolist())

    def get_tile(self, i: int) -> Tile:
        """The parsed (still compressed) tile ``i``, from the cache, an in-flight prefetch, or a fresh fetch."""
        with self._lock:
            fut = self._tiles.get(i)
            if fut is None:
                if not 0 <= i < self.n_tiles:
                    raise IndexError(f"tile {i} out of range")
                self._evict_tiles({i})
                fut = self._tiles[i] = self._io().submit(self._fetch_tile, i)
            else:
                self._tiles.move_to_end(i)
                if fut.done():
                    self.stats["tile_cache_hits"] += 1
        t0 = time.perf_counter()
        try:
            tile = fut.result()
        except BaseException:
            with self._lock:
                if self._tiles.get(i) is fut:
                    del self._tiles[i]  # let a later call retry
            raise
        with self._stats_lock:
            self.stats["stall_seconds"] += time.perf_counter() - t0
        return tile

    # ------------------------------------------------------------------ decoding

    def _acquire_buffers(self) -> tuple[np.ndarray, np.ndarray]:
        if self._free_buffers:
            return self._free_buffers.pop()
        cap = max(self.manifest.tile_nnz)
        return np.empty(cap, dtype=np.int32), np.empty(cap, dtype=np.int32)

    def _decoded_tile(self, i: int, tile: Tile) -> DecodedTile:
        hit = self._decoded.get(i)
        if hit is not None:
            self._decoded.move_to_end(i)
            self.stats["decoded_cache_hits"] += 1
            return hit[0]
        while len(self._decoded) >= self.max_cached_tiles:
            _, (_, bufs) = self._decoded.popitem(last=False)
            self._free_buffers.append(bufs)
        bufs = self._acquire_buffers()
        threads = min(self.decode_threads, max(1, tile.n_chunks))
        if len(self._scratch) < threads * tile.info.scratch_nbytes:
            self._scratch = np.empty(threads * tile.info.scratch_nbytes, dtype=np.uint8)
        if threads > 1 and self._decode_pool is None:
            self._decode_pool = ThreadPoolExecutor(self.decode_threads, thread_name_prefix="deltacells-decode")
        t0 = time.perf_counter()
        try:
            dec = tile.decode(
                values=np.int32,
                threads=threads,
                out_indices=bufs[0],
                out_values=bufs[1],
                scratch=self._scratch,
                executor=self._decode_pool,
            )
        except BaseException:
            self._free_buffers.append(bufs)
            raise
        self.stats["tiles_decoded"] += 1
        self.stats["decode_seconds"] += time.perf_counter() - t0
        self._decoded[i] = (dec, bufs)
        return dec

    # ------------------------------------------------------------------ batches

    def _layout(self, idx: np.ndarray):
        """Fetch the tiles ``idx`` touches (parsed only) and compute where each requested cell goes in the output."""
        tile_ids = np.searchsorted(self._limits, idx, side="right")
        order = list(dict.fromkeys(tile_ids.tolist()))
        self.prefetch(order)
        tiles = {t: self.get_tile(t) for t in order}
        local = idx - self._starts[tile_ids]
        lens = np.empty(len(idx), dtype=np.int64)
        for t in order:
            m = tile_ids == t
            ip = tiles[t].indptr
            lens[m] = ip[local[m] + 1] - ip[local[m]]
        out_indptr = np.zeros(len(idx) + 1, dtype=np.int64)
        np.cumsum(lens, out=out_indptr[1:])
        return tile_ids, order, tiles, local, out_indptr

    def _as_indices(self, indices: Sequence[int] | np.ndarray) -> np.ndarray:
        idx = np.ascontiguousarray(np.asarray(indices, dtype=np.int64)).ravel()
        if len(idx) and (idx.min() < 0 or idx.max() >= self.n_cells):
            raise IndexError(f"cell index out of range [0, {self.n_cells})")
        return idx

    def batch_nnz(self, indices: Sequence[int] | np.ndarray) -> int:
        """Number of nonzeros :meth:`get_batch` would return for ``indices`` (fetches the tiles but does not decode them).

        Use it to allocate exactly-sized output buffers (e.g. shared-memory tensors) before calling ``get_batch``.
        """
        idx = self._as_indices(indices)
        with self._lock:
            return int(self._layout(idx)[4][-1])

    def get_batch(
        self,
        indices: Sequence[int] | np.ndarray,
        *,
        values_dtype: Any = np.float32,
        out_indices: np.ndarray | None = None,
        out_values: np.ndarray | None = None,
    ) -> SparseBatch:
        """The cells with the given global indices, in the given order, as a CSR batch.

        Args:
            indices: Global cell indices (repeats allowed).
            values_dtype: ``float32`` (default) or ``int32``.
            out_indices, out_values: Optional preallocated 1-D buffers (int32 / ``values_dtype``) of at least as many elements as
                the batch has nonzeros (see :meth:`batch_nnz`) -- e.g. views of shared-memory tensors -- to receive the result
                without another copy.

        Cells are gathered tile by tile; a batch spanning several tiles decodes each of them (use :meth:`prefetch_cells` ahead of
        time and iterate cells so that consecutive batches share tiles).
        """
        idx = self._as_indices(indices)
        vdt = as_values_dtype(values_dtype)
        with self._lock:
            tile_ids, order, tiles, local, out_indptr = self._layout(idx)
            nnz = int(out_indptr[-1])
            if out_indices is None:
                out_indices = np.empty(nnz, dtype=np.int32)
            else:
                _check_out(out_indices, "out_indices", nnz, np.dtype(np.int32))
            if out_values is None:
                out_values = np.empty(nnz, dtype=vdt)
            else:
                _check_out(out_values, "out_values", nnz, vdt)
            for t in order:
                sel = np.nonzero(tile_ids == t)[0]
                dec = self._decoded_tile(t, tiles[t])
                gather_into(
                    dec.indptr,
                    dec.indices,
                    dec.values,
                    np.ascontiguousarray(local[sel]),
                    np.ascontiguousarray(out_indptr[sel]),
                    out_indices,
                    out_values,
                )
        return SparseBatch(out_indptr, out_indices[:nnz], out_values[:nnz], self.n_genes)

    def __getitem__(self, key: int | slice | Sequence[int] | np.ndarray) -> SparseBatch:
        if isinstance(key, (int, np.integer)):
            k = int(key)
            return self.get_batch([k + self.n_cells if k < 0 else k])
        if isinstance(key, slice):
            return self.get_batch(np.arange(*key.indices(self.n_cells)))
        return self.get_batch(key)


def open_dataset(uri: str | Backend, **kwargs: Any) -> DeltaCellsDataset:
    """Open a dataset directory / URI. See :class:`DeltaCellsDataset` for the options."""
    return DeltaCellsDataset(uri, **kwargs)
