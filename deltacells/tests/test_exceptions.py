# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""Counts and gene gaps that do not fit in a uint16 are stored as exceptions: exactness, limits and corrupt-frame handling."""

import struct

import numpy as np
import pytest
import scipy.sparse as sp
from conftest import make_counts, same

from deltacells import Tile, _core, encode_tile
from deltacells.format import (
    CHUNK_ENTRY_SIZE,
    HEADER_SIZE,
    MAX_VALUE,
    STREAM_MAX,
    FormatError,
    exception_limit,
    parse_tile_info,
)


def csr(entries, shape):
    rows, cols, vals = zip(*entries)
    m = sp.csr_matrix((vals, (rows, cols)), shape=shape, dtype=np.int64)
    m.sort_indices()
    return m


def n_value_frames(blob) -> int:
    return sum(c.vexc_len > 0 for c in parse_tile_info(blob).chunks)


def n_gap_frames(blob) -> int:
    return sum(c.gexc_len > 0 for c in parse_tile_info(blob).chunks)


def check_decodes(blob, m, threads=(1, 3)):
    t = Tile(blob)
    for n in threads:
        d = t.decode(values=np.int32, threads=n)
        assert same(d.to_scipy(), m)
        f = t.decode(values=np.float32, threads=n)  # exact as long as the values are exactly representable as float32
        assert np.array_equal(f.values, m.data.astype(np.float32)) and np.array_equal(f.indices, m.indices)
    rows = np.random.default_rng(0).permutation(m.shape[0])
    assert same(t.decode(values=np.int32).gather(rows, values_dtype=np.int32).to_scipy(), m[rows])


def test_a_tile_without_exceptions_has_none():
    blob = encode_tile(make_counts(200, 100, density=0.2, seed=1), n_chunks=4, level=3)
    assert n_value_frames(blob) == 0 and n_gap_frames(blob) == 0


def test_the_largest_uint16_is_not_an_exception_but_the_next_value_is():
    m = csr([(0, 0, STREAM_MAX), (1, 3, 5)], (2, 10))
    blob = encode_tile(m, level=3)
    assert n_value_frames(blob) == 0
    check_decodes(blob, m)
    m = csr([(0, 0, STREAM_MAX + 1), (1, 3, 5)], (2, 10))
    blob = encode_tile(m, level=3)
    assert n_value_frames(blob) == 1
    check_decodes(blob, m)


def test_the_largest_uint16_gap_is_not_an_exception_but_the_next_gap_is():
    m = csr([(0, 0, 1), (0, STREAM_MAX, 2), (1, 7, 3)], (2, 70_000))  # a gap of 65535
    blob = encode_tile(m, level=3)
    assert n_gap_frames(blob) == 0
    check_decodes(blob, m)
    m = csr([(0, 0, 1), (0, STREAM_MAX + 1, 2), (1, 7, 3)], (2, 70_000))  # a gap of 65536
    blob = encode_tile(m, level=3)
    assert n_gap_frames(blob) == 1
    check_decodes(blob, m)


@pytest.mark.parametrize("n_chunks", [1, 4, 9])
def test_large_values_roundtrip_exactly(n_chunks):
    rng = np.random.default_rng(0)
    m = make_counts(300, 80, density=0.2, seed=2, dtype=np.int64)
    big = rng.choice(m.nnz, size=60, replace=False)
    m.data[big] = rng.integers(STREAM_MAX + 1, 2**24, size=60)  # below 2**24, so that float32 holds them exactly
    m.data[big[:3]] = [MAX_VALUE >> 8, 2**24 - 1, 2**20]
    blob = encode_tile(m, n_chunks=n_chunks, level=3)
    assert n_value_frames(blob) >= 1
    check_decodes(blob, m)


def test_the_largest_value_roundtrips_as_int32():
    m = csr([(0, 0, MAX_VALUE), (0, 5, 70_000), (1, 9, MAX_VALUE), (2, 1, 3)], (3, 10))
    t = Tile(encode_tile(m, n_chunks=2, level=3))
    assert same(t.decode(values=np.int32).to_scipy(), m)


def test_large_gaps_roundtrip_exactly():
    entries = [
        (0, 0, 1), (0, 150_000, 2),  # a gap of 150000 after the first gene
        (1, 100_000, 3),  # the first gap (the gene index) is large
        (2, 0, 4), (2, 70_000, 5), (2, 140_000, 6), (2, 199_999, 7),  # three gaps, two of them large, in one cell
        (3, 5, 1), (3, 6, 1), (3, 7, 1),  # nothing special
        (5, 199_999, 9),  # no cell 4: it is empty
    ]  # fmt: skip
    m = csr(entries, (6, 200_000))
    for n_chunks in (1, 2, 5):
        blob = encode_tile(m, n_chunks=n_chunks, level=3)
        assert n_gap_frames(blob) >= 1
        check_decodes(blob, m)


def test_large_values_and_large_gaps_together():
    m = csr([(0, 0, 70_000), (0, 120_000, 2), (1, 130_000, 200_000), (1, 130_001, 1)], (2, 140_000))
    blob = encode_tile(m, n_chunks=2, level=3)
    assert n_value_frames(blob) >= 1 and n_gap_frames(blob) >= 1
    check_decodes(blob, m)


def test_fuzz_exceptions():
    rng = np.random.default_rng(7)
    for _ in range(40):
        n_cells, n_genes = int(rng.integers(1, 100)), int(rng.integers(70_000, 150_000))
        rows, cols, vals = [], [], []
        for r in range(n_cells):
            genes = np.unique(
                rng.integers(0, n_genes, size=int(rng.poisson(5)))
            )  # gaps of ~n_genes / 5: some are > 65535
            rows += [r] * len(genes)
            cols += genes.tolist()
            vals += np.where(
                rng.random(len(genes)) < 0.1, rng.integers(65_536, 2**24, len(genes)), rng.integers(1, 50, len(genes))
            ).tolist()
        if not rows:
            continue
        m = csr(list(zip(rows, cols, vals)), (n_cells, n_genes))
        blob = encode_tile(
            m, n_chunks=int(rng.integers(1, 8)), level=int(rng.choice([1, 3])), threads=int(rng.integers(1, 4))
        )
        check_decodes(blob, m, threads=(int(rng.integers(1, 5)),))


# ------------------------------------------------------------------------------------------------------ the limits


def test_a_tile_accepts_up_to_the_limit_of_value_exceptions_and_refuses_more():
    n = exception_limit(2 * 1024)  # 1024: there are two nonzeros per cell
    assert n == 1024

    def tile(n_cells):
        return csr([(r, c, 1 if c == 0 else 70_000) for r in range(n_cells) for c in (0, 1)], (n_cells, 2))

    check_decodes(encode_tile(tile(n), level=1), tile(n), threads=(2,))
    with pytest.raises(ValueError, match="does not look like raw counts"):
        encode_tile(tile(n + 1), level=1)


def test_a_tile_accepts_up_to_the_limit_of_gap_exceptions_and_refuses_more():
    n = exception_limit(2 * 1024)

    def tile(n_cells):  # every cell has a gap of 66000
        return csr([(r, c, 1) for r in range(n_cells) for c in (0, 66_000)], (n_cells, 70_000))

    check_decodes(encode_tile(tile(n), level=1), tile(n), threads=(2,))
    with pytest.raises(ValueError, match="too sparse"):
        encode_tile(tile(n + 1), level=1)


def test_the_limit_grows_with_the_tile():
    assert exception_limit(10) == 1024 and exception_limit(20_000_000) == 100_000


def test_values_above_the_largest_int32_are_rejected():
    m = csr([(0, 0, MAX_VALUE + 1)], (1, 3))
    with pytest.raises(ValueError, match="values must lie"):
        encode_tile(m)


# ---------------------------------------------------------------------------------------- corrupt exception frames


def apply(raw: bytes, kind: int, *, n: int = 10, n_genes: int = 100_000, level: int = 3):
    """Run apply_exceptions on a crafted frame over a chunk of ``n`` nonzeros in one cell, whose gene indices are 0..n-1."""
    frame = _core.compress_bytes(raw, level) if raw is not None else b"not a zstd frame"
    indptr = np.array([0, n], dtype=np.int64)
    out_idx = np.arange(n, dtype=np.int32)
    out_val = np.ones(n, dtype=np.int32)
    _core.apply_exceptions(frame, 0, len(frame), kind, 0, 1, 0, n, n_genes, indptr, out_idx, out_val, False)
    return out_idx, out_val


def u32(*values) -> bytes:
    return np.array(values, dtype="<u4").tobytes()


def test_apply_exceptions_patches_values_and_shifts_gene_indices():
    _, vals = apply(u32(2, 5, 70_000, 2**31 - 1), 0)
    assert vals.tolist() == [1, 1, 70_000, 1, 1, 2**31 - 1, 1, 1, 1, 1]
    idx, _ = apply(u32(4, 7, 65_536, 1), 1, n_genes=200_000)  # the indices from nonzero 4 on, then from 7 on, shift
    assert idx.tolist() == [0, 1, 2, 3, 4 + 65_536, 5 + 65_536, 6 + 65_536, 7 + 65_537, 8 + 65_537, 9 + 65_537]


@pytest.mark.parametrize(
    "raw,kind,message",
    [
        (u32(3, 3, 70_000, 70_000), 0, "positions"),  # not increasing
        (u32(5, 4, 70_000, 70_000), 0, "positions"),
        (u32(10, 70_000), 0, "positions"),  # out of the chunk
        (u32(0, 65_535), 0, "value out of range"),  # would fit in a uint16
        (u32(0, 2**31), 0, "value out of range"),  # more than an int32
        (u32(0, 0), 1, "value out of range"),  # a gap excess is at least 1
        (u32(0, 100_000), 1, "value out of range"),  # and less than n_genes
        (u32(0, 99_995), 1, "gene index out of range"),  # the cell would have a gene index >= n_genes
        (b"\x00" * 12, 0, "bad size"),  # not a whole number of (position, payload) pairs
        (u32(*range(11), *[70_000] * 11), 0, "bad size"),  # more exceptions than nonzeros
        (None, 0, "bad size"),  # not a zstd frame
    ],
)
def test_corrupt_exception_frames_are_rejected(raw, kind, message):
    with pytest.raises(ValueError, match=message):
        apply(raw, kind)


def test_an_exception_frame_outside_the_blob_is_rejected():
    m = csr([(0, 0, 70_000), (1, 3, 5)], (2, 10))
    buf = bytearray(encode_tile(m, n_chunks=1, level=3))
    struct.pack_into("<Q", buf, HEADER_SIZE + 9 * 8, 10**9)  # vexc_len of the first chunk
    assert CHUNK_ENTRY_SIZE == 96
    with pytest.raises(FormatError, match="outside"):
        parse_tile_info(buf)
