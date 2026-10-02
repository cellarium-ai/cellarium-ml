# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
The deltacells tile format (version 1). See ``docs/FORMAT.md`` for the full specification.

A tile is one self-contained blob holding a block of cells (rows) of a count matrix in CSR form::

    header (64 bytes) | chunk table (n_chunks * 64 bytes) | indptr (int64 * (n_cells + 1)) | compressed streams

Within each chunk of consecutive cells, the nonzeros are stored as two zstd-compressed streams of uint16 values
(byte-planar: all low bytes, then all high bytes): the per-cell *gaps* between consecutive gene indices (the first
gap of a cell is its first gene index) and the counts. All integers are little-endian.
"""

import struct
import sys
import zlib
from dataclasses import dataclass

MAGIC = b"DELTACEL"
VERSION = 1
HEADER_SIZE = 64
CHUNK_ENTRY_SIZE = 64
MAX_GENES = 65536  # gaps and gene indices are stored as uint16
MAX_VALUE = 65535  # counts are stored as uint16

# magic, version, flags, n_cells, nnz, n_genes, n_chunks, indptr_off, total_size, crc32, reserved
_HEADER = struct.Struct("<8sIIQQIIQQII")
# cell_lo, cell_hi, nnz_lo, nnz_hi, gaps_off, gaps_len, vals_off, vals_len
_CHUNK = struct.Struct("<8Q")
assert _HEADER.size == HEADER_SIZE and _CHUNK.size == CHUNK_ENTRY_SIZE

if sys.byteorder != "little":  # pragma: no cover
    raise ImportError("deltacells requires a little-endian host")


class FormatError(ValueError):
    """The blob is not a valid deltacells tile (wrong magic/version, truncated, inconsistent tables, ...)."""


@dataclass(frozen=True)
class ChunkInfo:
    cell_lo: int
    cell_hi: int
    nnz_lo: int
    nnz_hi: int
    gaps_off: int
    gaps_len: int
    vals_off: int
    vals_len: int

    @property
    def nnz(self) -> int:
        return self.nnz_hi - self.nnz_lo


@dataclass(frozen=True)
class TileInfo:
    version: int
    flags: int
    n_cells: int
    nnz: int
    n_genes: int
    indptr_off: int
    total_size: int
    crc32: int
    chunks: tuple[ChunkInfo, ...]

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    @property
    def max_chunk_nnz(self) -> int:
        return max((c.nnz for c in self.chunks), default=0)

    @property
    def scratch_nbytes(self) -> int:
        """Scratch bytes one decoding thread needs (two byte-planar uint16 streams of the largest chunk)."""
        return 4 * self.max_chunk_nnz


def align8(n: int) -> int:
    return (n + 7) & ~7


def pack_header(
    *, n_cells: int, nnz: int, n_genes: int, n_chunks: int, indptr_off: int, total_size: int, crc32: int
) -> bytes:
    return _HEADER.pack(MAGIC, VERSION, 0, n_cells, nnz, n_genes, n_chunks, indptr_off, total_size, crc32, 0)


def pack_chunk(chunk: ChunkInfo) -> bytes:
    return _CHUNK.pack(
        chunk.cell_lo,
        chunk.cell_hi,
        chunk.nnz_lo,
        chunk.nnz_hi,
        chunk.gaps_off,
        chunk.gaps_len,
        chunk.vals_off,
        chunk.vals_len,
    )


def parse_tile_info(buf) -> TileInfo:
    """Parse and validate the header and chunk table of a tile blob (any bytes-like object)."""
    mv = memoryview(buf)
    if mv.ndim != 1 or mv.itemsize != 1:
        mv = mv.cast("B")
    size = len(mv)
    if size < HEADER_SIZE:
        raise FormatError("blob is smaller than the tile header")
    magic, version, flags, n_cells, nnz, n_genes, n_chunks, indptr_off, total_size, crc32, _ = _HEADER.unpack_from(
        mv, 0
    )
    if magic != MAGIC:
        raise FormatError("bad magic: not a deltacells tile")
    if version != VERSION:
        raise FormatError(f"unsupported tile format version {version} (this build reads version {VERSION})")
    if flags != 0:
        raise FormatError(f"unsupported flags {flags:#x}")
    if total_size != size:
        raise FormatError(f"size mismatch: header says {total_size} bytes, blob has {size} (truncated or padded?)")
    if n_genes == 0 or n_genes > MAX_GENES:
        raise FormatError(f"invalid n_genes {n_genes}")
    table_end = HEADER_SIZE + n_chunks * CHUNK_ENTRY_SIZE
    indptr_end = indptr_off + 8 * (n_cells + 1)
    if table_end > size or indptr_off % 8 != 0 or indptr_off < table_end or indptr_end > size:
        raise FormatError("inconsistent chunk table / indptr offsets")
    chunks = []
    cell, pos = 0, 0
    for i in range(n_chunks):
        c = ChunkInfo(*_CHUNK.unpack_from(mv, HEADER_SIZE + i * CHUNK_ENTRY_SIZE))
        if c.cell_lo != cell or c.cell_hi < c.cell_lo or c.nnz_lo != pos or c.nnz_hi < c.nnz_lo:
            raise FormatError(f"chunk {i} is not contiguous with the previous chunk")
        for off, length in ((c.gaps_off, c.gaps_len), (c.vals_off, c.vals_len)):
            if off % 8 != 0 or off < indptr_end or off > size or length > size - off:
                raise FormatError(f"chunk {i} stream lies outside the blob")
        cell, pos = c.cell_hi, c.nnz_hi
        chunks.append(c)
    if cell != n_cells or pos != nnz:
        raise FormatError("chunks do not cover all cells / nonzeros")
    return TileInfo(version, flags, n_cells, nnz, n_genes, indptr_off, total_size, crc32, tuple(chunks))


def payload_crc32(buf) -> int:
    """CRC32 of everything after the 64-byte header."""
    return zlib.crc32(memoryview(buf)[HEADER_SIZE:]) & 0xFFFFFFFF
