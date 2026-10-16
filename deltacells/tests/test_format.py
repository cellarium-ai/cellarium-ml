# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from conftest import make_counts

import deltacells
from deltacells import FormatError, Tile, encode_tile
from deltacells.format import CHUNK_ENTRY_SIZE, HEADER_SIZE, MAGIC, parse_tile_info


@pytest.fixture
def blob():
    return encode_tile(make_counts(200, 80, seed=1), n_chunks=4, level=3)


def test_header_layout(blob):
    assert blob[:8] == MAGIC
    info = parse_tile_info(blob)
    assert info.n_cells == 200 and info.n_genes == 80 and info.n_chunks == 4
    assert info.total_size == len(blob)
    assert info.indptr_off % 8 == 0 and all(c.gaps_off % 8 == 0 and c.vals_off % 8 == 0 for c in info.chunks)
    assert info.indptr_off >= HEADER_SIZE + 4 * CHUNK_ENTRY_SIZE
    # chunks tile the cells and nonzeros contiguously
    assert info.chunks[0].cell_lo == 0 and info.chunks[-1].cell_hi == 200
    assert all(a.cell_hi == b.cell_lo and a.nnz_hi == b.nnz_lo for a, b in zip(info.chunks, info.chunks[1:]))


def test_bad_magic(blob):
    with pytest.raises(FormatError, match="magic"):
        Tile(b"XXXXXXXX" + blob[8:])


def test_unsupported_version(blob):
    b = bytearray(blob)
    b[8] = 99
    with pytest.raises(FormatError, match="version"):
        Tile(bytes(b))


def test_nonzero_flags_rejected(blob):
    b = bytearray(blob)
    b[12] = 1
    with pytest.raises(FormatError, match="flags"):
        Tile(bytes(b))


@pytest.mark.parametrize("cut", [0, 10, HEADER_SIZE, HEADER_SIZE + 10, -1, -100])
def test_truncated(blob, cut):
    with pytest.raises(FormatError):
        Tile(blob[:cut])


def test_trailing_garbage_rejected(blob):
    with pytest.raises(FormatError, match="size mismatch"):
        Tile(blob + b"\x00" * 8)


def test_non_contiguous_chunk_table_rejected(blob):
    b = bytearray(blob)
    # second chunk's cell_lo += 1
    off = HEADER_SIZE + CHUNK_ENTRY_SIZE
    b[off : off + 8] = (int.from_bytes(b[off : off + 8], "little") + 1).to_bytes(8, "little")
    with pytest.raises(FormatError, match="contiguous"):
        Tile(bytes(b))


def test_stream_offset_outside_blob_rejected(blob):
    b = bytearray(blob)
    off = HEADER_SIZE + 32  # first chunk gaps_off
    b[off : off + 8] = (len(blob) + 8).to_bytes(8, "little")
    with pytest.raises(FormatError, match="outside"):
        Tile(bytes(b))


def test_inconsistent_indptr_rejected(blob):
    info = parse_tile_info(blob)
    b = bytearray(blob)
    off = info.indptr_off + 8 * 5
    b[off : off + 8] = (10**9).to_bytes(8, "little")
    with pytest.raises(FormatError):
        Tile(bytes(b))


def test_crc_detects_payload_corruption(blob):
    Tile(blob, verify=True)  # intact: fine
    info = parse_tile_info(blob)
    b = bytearray(blob)
    b[info.chunks[1].vals_off + 3] ^= 0xFF
    with pytest.raises(FormatError, match="CRC"):
        Tile(bytes(b), verify=True)


def test_garbage_stream_raises_not_crashes(blob):
    info = parse_tile_info(blob)
    b = bytearray(blob)
    c = info.chunks[2]
    b[c.gaps_off : c.gaps_off + c.gaps_len] = bytes(
        np.random.default_rng(0).integers(0, 256, c.gaps_len, dtype=np.uint8)
    )
    tile = Tile(bytes(b))  # structurally valid
    with pytest.raises(ValueError, match="corrupt"):
        tile.decode()


def test_wrong_decompressed_size_detected():
    # a valid zstd frame of the wrong length in place of a stream
    pytest.importorskip("numcodecs")
    from numcodecs import Zstd

    blob = bytearray(encode_tile(make_counts(40, 30, seed=2), n_chunks=1, level=3))
    info = parse_tile_info(bytes(blob))
    c = info.chunks[0]
    bad = Zstd(3).encode(b"\x01" * 7)
    assert len(bad) <= c.gaps_len
    blob[c.gaps_off : c.gaps_off + len(bad)] = bad
    with pytest.raises(ValueError, match="corrupt"):
        Tile(bytes(blob)).decode()


def test_public_names():
    for name in deltacells.__all__:
        assert hasattr(deltacells, name)
