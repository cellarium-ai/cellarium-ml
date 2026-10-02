# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""Reading tiles: parsing, decoding into caller-owned buffers, and gathering batches of cells."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp

from deltacells import _core
from deltacells.format import FormatError, TileInfo, parse_tile_info, payload_crc32

if TYPE_CHECKING:  # pragma: no cover
    import torch


def as_values_dtype(dtype: Any) -> np.dtype:
    """Normalize a requested value dtype; only int32 (raw counts) and float32 are supported."""
    dt = np.dtype(dtype)
    if dt not in (np.dtype(np.int32), np.dtype(np.float32)):
        raise ValueError(f"values dtype must be int32 or float32, got {dt}")
    return dt


def _check_out(arr: np.ndarray, name: str, n: int, dtype: np.dtype) -> None:
    if (
        not isinstance(arr, np.ndarray)
        or arr.dtype != dtype
        or arr.ndim != 1
        or not arr.flags.c_contiguous
        or not arr.flags.writeable
    ):
        raise ValueError(f"{name} must be a writable contiguous 1-D {dtype} numpy array")
    if len(arr) < n:
        raise ValueError(f"{name} holds {len(arr)} elements but {n} are required")


@dataclass
class SparseBatch:
    """A block of cells in CSR form: ``indptr`` (int64, length n_rows + 1), ``indices`` (int32), ``values`` (float32 or int32)."""

    indptr: np.ndarray
    indices: np.ndarray
    values: np.ndarray
    n_genes: int

    @property
    def n_rows(self) -> int:
        return len(self.indptr) - 1

    @property
    def nnz(self) -> int:
        return int(self.indptr[-1])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_rows, self.n_genes)

    def to_scipy(self) -> sp.csr_matrix:
        return sp.csr_matrix((self.values[: self.nnz], self.indices[: self.nnz], self.indptr), shape=self.shape)

    def to_torch_csr(self) -> torch.Tensor:
        """A ``torch.sparse_csr`` tensor sharing memory with ``indices`` and ``values`` (``indptr`` is converted to int32)."""
        import torch

        if self.nnz >= 2**31:
            raise OverflowError("batch has too many nonzeros for int32 CSR indices")
        return torch.sparse_csr_tensor(
            torch.from_numpy(self.indptr.astype(np.int32)),
            torch.from_numpy(self.indices[: self.nnz]),
            torch.from_numpy(self.values[: self.nnz]),
            size=self.shape,
            check_invariants=False,
        )


def gather_into(
    indptr: np.ndarray,
    indices: np.ndarray,
    values: np.ndarray,
    rows: np.ndarray,
    out_offsets: np.ndarray,
    out_indices: np.ndarray,
    out_values: np.ndarray,
) -> None:
    """Copy ``rows`` of a CSR matrix to ``out_*[out_offsets[i]:...]``. Values are converted int32 -> float32 if the dtypes differ."""
    if values.dtype == out_values.dtype:
        convert = False
    elif values.dtype == np.int32 and out_values.dtype == np.float32:
        convert = True
    else:
        raise ValueError(f"cannot convert {values.dtype} values to {out_values.dtype}")
    _core.gather_rows(indptr, indices, values, rows, out_offsets, out_indices, out_values, convert)


def row_offsets(indptr: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Output ``indptr`` (int64, length len(rows) + 1) for gathering ``rows``."""
    if len(rows) and (rows.min() < 0 or rows.max() >= len(indptr) - 1):
        raise IndexError("row index out of range")
    lens = indptr[rows + 1] - indptr[rows]
    out = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum(lens, out=out[1:])
    return out


@dataclass
class DecodedTile:
    """A fully decoded tile in CSR form."""

    indptr: np.ndarray  # int64, n_cells + 1
    indices: np.ndarray  # int32, nnz
    values: np.ndarray  # int32 or float32, nnz
    n_genes: int

    @property
    def n_cells(self) -> int:
        return len(self.indptr) - 1

    @property
    def nnz(self) -> int:
        return int(self.indptr[-1])

    def to_scipy(self) -> sp.csr_matrix:
        return sp.csr_matrix((self.values, self.indices, self.indptr), shape=(self.n_cells, self.n_genes))

    def gather(
        self,
        rows: Sequence[int] | np.ndarray,
        *,
        values_dtype: Any = np.float32,
        out_indices: np.ndarray | None = None,
        out_values: np.ndarray | None = None,
    ) -> SparseBatch:
        """Select ``rows`` (in the given order, repeats allowed) into a new CSR batch, optionally into caller-provided buffers."""
        rows = np.ascontiguousarray(rows, dtype=np.int64)
        vdt = as_values_dtype(values_dtype)
        out_indptr = row_offsets(self.indptr, rows)
        nnz = int(out_indptr[-1])
        if out_indices is None:
            out_indices = np.empty(nnz, dtype=np.int32)
        else:
            _check_out(out_indices, "out_indices", nnz, np.dtype(np.int32))
        if out_values is None:
            out_values = np.empty(nnz, dtype=vdt)
        else:
            _check_out(out_values, "out_values", nnz, vdt)
        gather_into(self.indptr, self.indices, self.values, rows, out_indptr[:-1].copy(), out_indices, out_values)
        return SparseBatch(out_indptr, out_indices[:nnz], out_values[:nnz], self.n_genes)


class Tile:
    """A parsed tile blob. Holds a reference to the (bytes-like) blob; nothing is decompressed until :meth:`decode`.

    Args:
        data: The tile bytes (``bytes``, ``bytearray``, ``memoryview``, a uint8 ``ndarray``, an ``mmap`` ...).
        verify: Also check the payload CRC32 (reads the whole blob once).
    """

    def __init__(self, data, *, verify: bool = False) -> None:
        self._data = data
        self._mv = memoryview(data)
        if self._mv.ndim != 1 or self._mv.itemsize != 1:
            self._mv = self._mv.cast("B")
        self.info: TileInfo = parse_tile_info(self._mv)
        if verify:
            self.verify()
        # an aligned native-endian copy (80 bytes per 10 cells) so that C code never reads misaligned integers
        self.indptr = np.frombuffer(
            self._mv, dtype="<i8", count=self.info.n_cells + 1, offset=self.info.indptr_off
        ).astype(np.int64)
        if (
            self.indptr[0] != 0
            or self.indptr[-1] != self.info.nnz
            or (len(self.indptr) > 1 and np.any(np.diff(self.indptr) < 0))
        ):
            raise FormatError("invalid indptr")
        for c in self.info.chunks:
            if self.indptr[c.cell_lo] != c.nnz_lo or self.indptr[c.cell_hi] != c.nnz_hi:
                raise FormatError("indptr does not match the chunk table")

    @property
    def n_cells(self) -> int:
        return self.info.n_cells

    @property
    def n_genes(self) -> int:
        return self.info.n_genes

    @property
    def nnz(self) -> int:
        return self.info.nnz

    @property
    def n_chunks(self) -> int:
        return self.info.n_chunks

    @property
    def nbytes(self) -> int:
        """Size of the compressed blob in bytes."""
        return len(self._mv)

    def verify(self) -> None:
        """Check the payload CRC32; raises :class:`FormatError` on mismatch."""
        if payload_crc32(self._mv) != self.info.crc32:
            raise FormatError("payload CRC32 mismatch: the tile is corrupt")

    def decode(
        self,
        *,
        values: Any = np.int32,
        threads: int = 1,
        out_indices: np.ndarray | None = None,
        out_values: np.ndarray | None = None,
        scratch: np.ndarray | None = None,
        executor: Executor | None = None,
    ) -> DecodedTile:
        """Decompress the whole tile.

        Args:
            values: ``int32`` (raw counts) or ``float32``.
            threads: Decode up to this many chunks in parallel (the C core releases the GIL).
            out_indices, out_values, scratch: Optional preallocated buffers to avoid allocation and page faults. ``out_*`` need
                at least ``nnz`` elements (int32 / the requested dtype); ``scratch`` (uint8) needs ``threads * info.scratch_nbytes``.
            executor: Thread pool to use when ``threads > 1`` (a temporary one is created otherwise).
        """
        info = self.info
        vdt = as_values_dtype(values)
        if out_indices is None:
            out_indices = np.empty(info.nnz, dtype=np.int32)
        else:
            _check_out(out_indices, "out_indices", info.nnz, np.dtype(np.int32))
        if out_values is None:
            out_values = np.empty(info.nnz, dtype=vdt)
        else:
            _check_out(out_values, "out_values", info.nnz, vdt)
        if info.nnz == 0:
            return DecodedTile(self.indptr, out_indices[:0], out_values[:0], info.n_genes)

        n_threads = max(1, min(threads, info.n_chunks))
        need = n_threads * info.scratch_nbytes
        if scratch is None:
            scratch = np.empty(need, dtype=np.uint8)
        elif scratch.dtype != np.uint8 or len(scratch) < need or not scratch.flags.writeable:
            raise ValueError(f"scratch must be a writable uint8 array of at least {need} bytes")
        to_float = vdt == np.dtype(np.float32)
        per_thread = info.scratch_nbytes

        def run(t: int) -> None:
            buf = scratch[t * per_thread : (t + 1) * per_thread]
            for c in info.chunks[t::n_threads]:
                _core.decode_chunk(
                    self._mv, c.gaps_off, c.gaps_len, c.vals_off, c.vals_len, c.cell_lo, c.cell_hi, c.nnz_lo, c.nnz_hi,
                    info.n_genes, self.indptr, buf, out_indices, out_values, to_float,
                )  # fmt: skip

        if n_threads == 1:
            run(0)
        else:
            pool = executor or ThreadPoolExecutor(n_threads)
            try:
                for f in [pool.submit(run, t) for t in range(n_threads)]:
                    f.result()
            finally:
                if executor is None:
                    pool.shutdown()
        return DecodedTile(self.indptr, out_indices[: info.nnz], out_values[: info.nnz], info.n_genes)

    def to_scipy(self, *, values: Any = np.float32, threads: int = 1) -> sp.csr_matrix:
        return self.decode(values=values, threads=threads).to_scipy()


def read_tile(path: str, *, verify: bool = False) -> Tile:
    """Read a tile file from local disk."""
    return Tile(np.fromfile(path, dtype=np.uint8), verify=verify)
