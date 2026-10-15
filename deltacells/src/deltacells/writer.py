# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""Writing tiles and datasets."""

from __future__ import annotations

import os
import shutil
import tempfile
import zlib
from collections import deque
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp

from deltacells import _core
from deltacells.format import (
    CHUNK_ENTRY_SIZE,
    HEADER_SIZE,
    MAX_GENES,
    MAX_VALUE,
    STREAM_MAX,
    ChunkInfo,
    align8,
    exception_limit,
    pack_chunk,
    pack_header,
    parse_tile_info,
)
from deltacells.manifest import (
    DEFAULT_OBS_TILES_PER_FILE,
    GENE_ORDER_NAME,
    MANIFEST_NAME,
    VAR_NAMES_NAME,
    VAR_TABLE_NAME,
    Manifest,
)
from deltacells.sink import LocalSink, open_sink

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    from deltacells.obs import ObsSchema

DEFAULT_LEVEL = 19
DEFAULT_CHUNKS = 8


def permute_columns(matrix: sp.csr_matrix, gene_order: np.ndarray) -> sp.csr_matrix:
    """Reorder genes: output column ``j`` is input column ``gene_order[j]``. Returns a new matrix; the input is not modified."""
    gene_order = np.asarray(gene_order)
    n = matrix.shape[1]
    if gene_order.shape != (n,) or not np.array_equal(np.sort(gene_order), np.arange(n)):
        raise ValueError("gene_order must be a permutation of range(n_genes)")
    new_col = np.empty(n, dtype=np.int32)
    new_col[gene_order] = np.arange(n, dtype=np.int32)
    # copy data and indptr: sorting the indices below reorders `data` in place and must not touch the caller's matrix
    out = sp.csr_matrix((matrix.data.copy(), new_col[matrix.indices], matrix.indptr.copy()), shape=matrix.shape)
    out.sort_indices()
    return out


def _canonical_counts(matrix: Any) -> sp.csr_matrix:
    """Validate that ``matrix`` holds non-negative integer counts (at most ``MAX_VALUE``) and return it as a canonical CSR matrix."""
    csr = sp.csr_matrix(matrix)
    if not csr.has_canonical_format:
        csr = csr.copy()
        csr.sum_duplicates()
    n_genes = csr.shape[1]
    if not 1 <= n_genes <= MAX_GENES:
        raise ValueError(f"n_genes must be between 1 and {MAX_GENES}, got {n_genes}")
    if csr.nnz:
        data = csr.data
        if np.issubdtype(data.dtype, np.floating):
            if not np.all(np.isfinite(data)) or not np.array_equal(data, np.rint(data)):
                raise ValueError("values must be integer counts (found non-integer or non-finite values)")
        elif not np.issubdtype(data.dtype, np.integer) and data.dtype != np.bool_:
            raise ValueError(f"unsupported value dtype {data.dtype}")
        if data.min() < 0 or data.max() > MAX_VALUE:
            raise ValueError(f"values must lie in [0, {MAX_VALUE}] (got min {data.min()}, max {data.max()})")
        if np.any(data == 0):
            csr = csr.copy()
            csr.eliminate_zeros()
    return csr


def _chunk_bounds(indptr: np.ndarray, n_chunks: int) -> list[tuple[int, int]]:
    """Split the cells into at most ``n_chunks`` runs of consecutive cells with roughly equal numbers of nonzeros."""
    n_cells, nnz = len(indptr) - 1, int(indptr[-1])
    if n_cells == 0:
        return []
    k = max(1, min(n_chunks, n_cells))
    cuts = np.searchsorted(indptr, np.linspace(0, nnz, k + 1)[1:-1], side="left")
    bounds = np.unique(np.concatenate([[0], cuts, [n_cells]]))
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]


def _compress(array: np.ndarray, level: int) -> np.ndarray:
    out = np.empty(_core.compress_bound(len(array)), dtype=np.uint8)
    size = _core.compress_stream(np.ascontiguousarray(array), level, out)
    return out[:size]


def _exception_frame(index: np.ndarray, payload: np.ndarray, lo: int, hi: int, level: int) -> np.ndarray | None:
    """The exception frame of the nonzeros ``lo:hi``: ``index`` lists the (sorted) nonzeros that are exceptions, ``payload`` their
    payloads. ``None`` if there are none."""
    i0, i1 = np.searchsorted(index, [lo, hi])
    if i0 == i1:
        return None
    raw = np.concatenate([(index[i0:i1] - lo).astype(np.uint32), payload[i0:i1].astype(np.uint32)])
    return np.frombuffer(_core.compress_bytes(raw, level), dtype=np.uint8)


def encode_tile(
    matrix: Any, *, n_chunks: int = DEFAULT_CHUNKS, level: int = DEFAULT_LEVEL, threads: int | None = None
) -> bytes:
    """Encode a block of cells (rows of a count matrix) as one tile blob.

    Args:
        matrix: Anything ``scipy.sparse`` can turn into CSR (cells x genes) holding integer counts in ``[0, 2**31 - 1]``, with at
            most ``2**31`` genes. Duplicates are summed and explicit zeros dropped. Counts above 65535, and gaps between a cell's
            consecutive nonzero genes above 65535, are stored exactly as *exceptions*; they must be rare (at most ``max(1024, nnz /
            200)`` of each kind per tile), otherwise a ``ValueError`` is raised.
        n_chunks: Number of independently compressed chunks (parallel decode granularity). More chunks cost ~1% size for 8.
        level: zstd level. Higher is smaller and (here) also decodes slightly faster; it only costs compression time.
        threads: Threads used for compression (default: up to 8).
    """
    if n_chunks < 1:
        raise ValueError("n_chunks must be at least 1")
    csr = _canonical_counts(matrix)
    n_cells, n_genes = csr.shape
    nnz = csr.nnz
    indptr = csr.indptr.astype(np.int64)

    gaps = np.empty(nnz, dtype=np.int64)
    if nnz:
        idx = csr.indices.astype(np.int64)
        gaps[0] = idx[0]
        gaps[1:] = idx[1:] - idx[:-1]
        starts = indptr[:-1][indptr[:-1] < nnz]
        gaps[starts] = idx[starts]  # the first gap of a cell is its first gene index
    vals = csr.data.astype(np.int64)

    big_vals = np.flatnonzero(vals > STREAM_MAX)
    big_gaps = np.flatnonzero(gaps > STREAM_MAX)
    limit = exception_limit(nnz)
    if len(big_vals) > limit:
        raise ValueError(
            f"{len(big_vals)} of the {nnz} nonzeros are counts above {STREAM_MAX} (the most a tile accepts is {limit}); "
            "this does not look like raw counts, which is what the deltacells format is for"
        )
    if len(big_gaps) > limit:
        raise ValueError(
            f"{len(big_gaps)} of the {nnz} nonzeros follow a gap of more than {STREAM_MAX} genes (the most a tile accepts is "
            f"{limit}); the cells are too sparse for {n_genes} genes, which the deltacells format does not support"
        )
    gaps16 = np.minimum(gaps, STREAM_MAX).astype(np.uint16)
    vals16 = np.minimum(vals, STREAM_MAX).astype(np.uint16)

    bounds = _chunk_bounds(indptr, n_chunks)
    if any(indptr[b] - indptr[a] >= 2**32 for a, b in bounds):
        raise ValueError("a chunk has 2**32 or more nonzeros; use more chunks (n_chunks)")
    n_threads = threads if threads is not None else min(8, os.cpu_count() or 1)
    jobs = [(a, b, name, arr[indptr[a] : indptr[b]]) for a, b in bounds for name, arr in (("g", gaps16), ("v", vals16))]
    if n_threads > 1 and len(jobs) > 1:
        with ThreadPoolExecutor(n_threads) as pool:
            streams = list(pool.map(lambda j: _compress(j[3], level), jobs))
    else:
        streams = [_compress(j[3], level) for j in jobs]

    table_end = HEADER_SIZE + CHUNK_ENTRY_SIZE * len(bounds)
    indptr_off = align8(table_end)
    pos = indptr_off + 8 * (n_cells + 1)
    chunks = []
    placed: list[tuple[int, np.ndarray]] = []  # (offset, bytes) of every stream and exception frame
    for i, (a, b) in enumerate(bounds):
        lo, hi = int(indptr[a]), int(indptr[b])
        frames = [
            streams[2 * i],
            streams[2 * i + 1],
            _exception_frame(big_vals, vals[big_vals], lo, hi, level),
            _exception_frame(big_gaps, gaps[big_gaps] - STREAM_MAX, lo, hi, level),
        ]
        entries = []
        for frame in frames:
            if frame is None:
                entries += [0, 0]
                continue
            off = align8(pos)
            placed.append((off, frame))
            pos = off + len(frame)
            entries += [off, len(frame)]
        chunks.append(ChunkInfo(a, b, lo, hi, *entries))
    total = pos
    buf = bytearray(total)
    for i, c in enumerate(chunks):
        buf[HEADER_SIZE + i * CHUNK_ENTRY_SIZE : HEADER_SIZE + (i + 1) * CHUNK_ENTRY_SIZE] = pack_chunk(c)
    for off, frame in placed:
        buf[off : off + len(frame)] = frame.data
    buf[indptr_off : indptr_off + 8 * (n_cells + 1)] = indptr.tobytes()
    crc = zlib.crc32(memoryview(buf)[HEADER_SIZE:]) & 0xFFFFFFFF
    buf[:HEADER_SIZE] = pack_header(
        n_cells=n_cells,
        nnz=nnz,
        n_genes=n_genes,
        n_chunks=len(chunks),
        indptr_off=indptr_off,
        total_size=total,
        crc32=crc,
    )
    return bytes(buf)


def _atomic_write(path: str, data) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def write_tile(path: str, matrix: Any, **kwargs: Any) -> int:
    """Encode ``matrix`` (see :func:`encode_tile`) and write it to ``path`` atomically. Returns the file size in bytes."""
    blob = encode_tile(matrix, **kwargs)
    _atomic_write(path, blob)
    return len(blob)


def write_tile_file(
    root: str,
    index: int,
    matrix: Any,
    *,
    gene_order: np.ndarray | None = None,
    n_chunks: int = DEFAULT_CHUNKS,
    level: int = DEFAULT_LEVEL,
    threads: int | None = None,
) -> tuple[int, int, int]:
    """Reorder the genes of ``matrix`` by ``gene_order`` (if given), encode it and write tile ``index`` of the dataset ``root``.

    Returns ``(n_cells, nnz, n_bytes)``, which :meth:`DatasetWriter.commit_tile` takes. Different tiles can be written from
    different processes; the ``tiles`` directory must exist (a :class:`DatasetWriter` creates it).
    """
    csr = sp.csr_matrix(matrix)
    if gene_order is not None:
        csr = permute_columns(_canonical_counts(csr), gene_order)
    blob = encode_tile(csr, n_chunks=n_chunks, level=level, threads=threads)
    _atomic_write(os.path.join(root, Manifest.tile_name(index)), blob)
    return csr.shape[0], parse_tile_info(blob).nnz, len(blob)


class DatasetWriter:
    """Write a dataset directory tile by tile (``tiles/tile_000000.dct``, ... and ``manifest.json``).

    Args:
        root: Output directory, or a remote prefix such as ``gs://bucket/prefix``. For a remote output everything is written to
            a local scratch directory (``scratch_dir``) and each file is uploaded as soon as it is finished and then deleted,
            so the scratch space needed is only a few tiles, not the dataset; the manifest is uploaded last.
        n_genes: Number of genes (columns).
        tile_size: Cells per tile; every tile except the last must have exactly this many.
        gene_order: Optional permutation. Incoming matrices are in the *original* gene order; output column ``j`` is input
            gene ``gene_order[j]``. Sorting genes by decreasing total counts makes the data much more compressible.
        var_names: Optional gene names in the *original* order; stored in output order.
        var: Optional gene table (a DataFrame with one row per gene, in the *original* order; the index is kept), stored as
            ``var.parquet`` in output order. Needs ``pandas`` and ``pyarrow``.
        obs_schema: Enables per-cell metadata: an :class:`deltacells.obs.ObsSchema` (see ``ObsSchema.infer``). Every tile must
            then be added with its ``obs`` (a DataFrame with the schema's columns, or an already encoded Arrow table).
        obs_tiles_per_file: Tiles per obs parquet shard (one row group per tile).
        obs_compression_level: zstd level of the obs parquet files.
        n_chunks, level, threads: See :func:`encode_tile`.
        metadata: Free-form JSON-serializable dict stored in the manifest.
        overwrite: Delete the dataset that is at ``root`` first (right away, before anything is written). Only the files of a
            deltacells dataset (complete or not) are deleted: if ``root`` holds anything else this raises instead.
        inputs: Paths of the files the dataset is made from. Raises if one is inside ``root``, which ``overwrite`` would
            delete.
        scratch_dir: For a remote ``root``, where to make the local scratch directory (default: the system temp directory).
        filesystem: An fsspec filesystem to use for a remote ``root`` instead of the default for its protocol (for testing).
        upload_threads: Files uploaded at the same time to a remote ``root``.

    The manifest is written by :meth:`close` (or on leaving a ``with`` block without an exception); a directory or prefix
    without a manifest is an incomplete dataset. :attr:`root` is the local directory the files are written to (``root``
    itself for a local output, the scratch directory for a remote one): that is where :func:`write_tile_file` must write
    the tiles that :meth:`commit_tile` registers. If writing fails, call :meth:`abort` (``with`` does).
    """

    def __init__(
        self,
        root: str,
        *,
        n_genes: int,
        tile_size: int,
        gene_order: Sequence[int] | np.ndarray | None = None,
        var_names: Sequence[str] | None = None,
        var: pd.DataFrame | None = None,
        obs_schema: ObsSchema | None = None,
        obs_tiles_per_file: int = DEFAULT_OBS_TILES_PER_FILE,
        obs_compression_level: int = 9,
        n_chunks: int = DEFAULT_CHUNKS,
        level: int = DEFAULT_LEVEL,
        threads: int | None = None,
        metadata: dict | None = None,
        overwrite: bool = False,
        inputs: Sequence[str] = (),
        scratch_dir: str | None = None,
        filesystem: Any = None,
        upload_threads: int = 8,
    ) -> None:
        # check everything before the destination is touched, so that a mistake does not cost an existing dataset
        if tile_size < 1:
            raise ValueError("tile_size must be positive")
        if not 1 <= n_genes <= MAX_GENES:
            raise ValueError(f"n_genes must be between 1 and {MAX_GENES}")
        self.n_genes, self.tile_size = n_genes, tile_size
        self.gene_order = None if gene_order is None else np.asarray(gene_order, dtype=np.int64)
        if self.gene_order is not None and (
            self.gene_order.shape != (n_genes,) or not np.array_equal(np.sort(self.gene_order), np.arange(n_genes))
        ):
            raise ValueError("gene_order must be a permutation of range(n_genes)")
        if var_names is not None and len(var_names) != n_genes:
            raise ValueError("var_names must have n_genes entries")
        self.var_names = None if var_names is None else [str(v) for v in var_names]
        if var is not None and len(var) != n_genes:
            raise ValueError("var must have n_genes rows")
        self.var = var
        if obs_tiles_per_file < 1:
            raise ValueError("obs_tiles_per_file must be at least 1")

        self.output = root
        self._sink = open_sink(root, filesystem)
        self._sink.check(overwrite, list(inputs))
        self._sink.probe()  # before anything is deleted: if the output cannot be written, what is there stays
        self._sink.prepare(overwrite, list(inputs))
        self._scratch: str | None = None
        self._pool: ThreadPoolExecutor | None = None
        self._uploads: deque[Future] = deque()
        self._max_uploads = 4 * upload_threads
        self.n_published = 0  # files uploaded to a remote output so far
        if self._sink.local:
            self.root = root
        else:
            self._scratch = self.root = tempfile.mkdtemp(prefix="deltacells_stage_", dir=scratch_dir)
            self._pool = ThreadPoolExecutor(upload_threads)
        os.makedirs(os.path.join(self.root, "tiles"), exist_ok=True)

        self.obs_schema = obs_schema
        self._obs_writer = None
        if obs_schema is not None:
            from deltacells.obs import ObsWriter

            self._obs_writer = ObsWriter(
                self.root,
                obs_schema,
                tiles_per_file=obs_tiles_per_file,
                compression_level=obs_compression_level,
                publish=self._publish,
            )
        self.obs_tiles_per_file = obs_tiles_per_file
        self.n_chunks, self.level, self.threads, self.metadata = n_chunks, level, threads, dict(metadata or {})
        self._cells: list[int] = []
        self._nnz: list[int] = []
        self._bytes: list[int] = []
        self._closed = False

    @property
    def is_remote(self) -> bool:
        """Whether the dataset is uploaded to a remote prefix (and ``root`` is only a scratch directory)."""
        return not self._sink.local

    @property
    def pending_uploads(self) -> int:
        """Files that are waiting for, or in, their upload (always 0 for a local output)."""
        return len(self._uploads)

    def _publish(self, name: str) -> None:
        """A file of the dataset is finished in ``root``: upload it (in the background) and delete the local copy. Does
        nothing for a local output, whose files are already where they belong. Raises the error of an upload that failed."""
        if self._pool is None:
            return
        self._uploads.append(self._pool.submit(self._upload, name))
        while self._uploads and (self._uploads[0].done() or len(self._uploads) > self._max_uploads):
            self._uploads.popleft().result()

    def _upload(self, name: str) -> None:
        path = os.path.join(self.root, name)
        self._sink.put(path, name)
        os.remove(path)
        self.n_published += 1

    def _wait_for_uploads(self) -> None:
        while self._uploads:
            self._uploads.popleft().result()

    def _cleanup(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        if self._scratch is not None:
            shutil.rmtree(self._scratch, ignore_errors=True)
            self._scratch = None

    def abort(self) -> None:
        """Give up on the dataset. The scratch directory is deleted; so is what was written to a local output (the files of
        the dataset, and the directory if this created it). Nothing is deleted from a remote output: the files already
        uploaded stay there, without a manifest, which marks the dataset as incomplete."""
        self._closed = True
        self._uploads.clear()
        self._cleanup()
        if isinstance(self._sink, LocalSink):
            self._sink.discard()

    def add_tile(self, matrix: Any, obs: pd.DataFrame | Any | None = None) -> int:
        """Write the next tile (and its obs rows, if the writer has an ``obs_schema``); returns its index.

        ``obs`` is a DataFrame with one row per cell of the tile (its index is ignored) and the schema's columns, or an Arrow table
        already encoded with the schema.
        """
        if self._closed:
            raise RuntimeError("writer is closed")
        csr = sp.csr_matrix(matrix)
        if csr.shape[1] != self.n_genes:
            raise ValueError(f"matrix has {csr.shape[1]} genes, expected {self.n_genes}")
        n = csr.shape[0]
        self._check_next_tile(n)
        obs_table = self._obs_table(obs, n)  # may raise: before anything is written
        i = len(self._cells)
        _, nnz, n_bytes = write_tile_file(
            self.root,
            i,
            csr,
            gene_order=self.gene_order,
            n_chunks=self.n_chunks,
            level=self.level,
            threads=self.threads,
        )
        self._record_tile(n, nnz, n_bytes, obs_table)
        return i

    def commit_tile(
        self, index: int, n_cells: int, nnz: int, n_bytes: int, obs: pd.DataFrame | Any | None = None
    ) -> None:
        """Register tile ``index``, which :func:`write_tile_file` already wrote to ``root`` (for example in another process).

        The tiles must be committed in order. ``n_cells``, ``nnz`` and ``n_bytes`` are what :func:`write_tile_file` returned;
        ``obs`` is as in :meth:`add_tile`.
        """
        if self._closed:
            raise RuntimeError("writer is closed")
        if index != len(self._cells):
            raise ValueError(f"tiles must be committed in order: expected tile {len(self._cells)}, got {index}")
        self._check_next_tile(n_cells)
        self._record_tile(n_cells, nnz, n_bytes, self._obs_table(obs, n_cells))

    def _check_next_tile(self, n: int) -> None:
        if not 0 < n <= self.tile_size:
            raise ValueError(f"a tile must hold between 1 and tile_size={self.tile_size} cells, got {n}")
        if self._cells and self._cells[-1] != self.tile_size:
            raise ValueError("only the last tile may hold fewer than tile_size cells")

    def _obs_table(self, obs: pd.DataFrame | Any | None, n: int) -> Any:
        if (obs is None) != (self._obs_writer is None):
            raise ValueError(
                "obs must be given for every tile if (and only if) the writer was created with an obs_schema"
            )
        if obs is None:
            return None
        assert self.obs_schema is not None
        table = obs if hasattr(obs, "schema") else self.obs_schema.encode(obs)
        if table.num_rows != n:
            raise ValueError(f"obs has {table.num_rows} rows but the tile has {n} cells")
        return table

    def _record_tile(self, n: int, nnz: int, n_bytes: int, obs_table: Any) -> None:
        if obs_table is not None:
            self._obs_writer.add_tile(obs_table)
        self._publish(Manifest.tile_name(len(self._cells)))
        self._cells.append(n)
        self._nnz.append(nnz)
        self._bytes.append(n_bytes)

    def close(self) -> Manifest:
        """Write the manifest (and gene order / variable names), after everything else has been uploaded if the output is
        remote. Returns it."""
        if self._closed:
            raise RuntimeError("writer is already closed")
        if not self._cells:
            raise ValueError("no tiles were written")
        self._closed = True
        if self.gene_order is not None:
            np.save(os.path.join(self.root, GENE_ORDER_NAME), self.gene_order.astype(np.int32))
            self._publish(GENE_ORDER_NAME)
        if self.var_names is not None:
            names = self.var_names if self.gene_order is None else [self.var_names[j] for j in self.gene_order]
            if any("\n" in v for v in names):
                raise ValueError("variable names must not contain newlines")
            with open(os.path.join(self.root, VAR_NAMES_NAME), "w") as f:
                f.write("\n".join(names) + "\n")
            self._publish(VAR_NAMES_NAME)
        if self.var is not None:
            from deltacells.obs import write_var_table

            write_var_table(self.root, self.var if self.gene_order is None else self.var.iloc[self.gene_order])
            self._publish(VAR_TABLE_NAME)
        obs_fields: dict[str, Any] = {}
        if self._obs_writer is not None:
            _, fingerprint = self._obs_writer.close(self._cells)
            obs_fields = {
                "has_obs": True,
                "obs_tiles_per_file": self.obs_tiles_per_file,
                "obs_fingerprint": fingerprint,
                "obs_shard_bytes": self._obs_writer.shard_bytes,
                "obs_shard_sha256": self._obs_writer.shard_sha256,
            }
        manifest = Manifest(
            n_cells=sum(self._cells),
            n_genes=self.n_genes,
            tile_size=self.tile_size,
            tile_cells=self._cells,
            tile_nnz=self._nnz,
            tile_bytes=self._bytes,
            n_chunks=self.n_chunks,
            zstd_level=self.level,
            has_gene_order=self.gene_order is not None,
            has_var_names=self.var_names is not None,
            metadata=self.metadata,
            has_var_table=self.var is not None,
            **obs_fields,
        )
        self._wait_for_uploads()  # everything but the manifest is in place
        manifest_path = os.path.join(self.root, MANIFEST_NAME)
        _atomic_write(manifest_path, manifest.to_json().encode())
        if self._scratch is not None:
            self._sink.put(manifest_path, MANIFEST_NAME)  # last, so that an incomplete dataset has no manifest
            self.n_published += 1
        self._cleanup()
        return manifest

    def __enter__(self) -> DatasetWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.abort()
        elif not self._closed:
            try:
                self.close()
            except BaseException:
                self.abort()
                raise
