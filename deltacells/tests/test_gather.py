# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from conftest import make_counts, same

from deltacells import Tile, encode_tile
from deltacells.reader import row_offsets


@pytest.fixture(scope="module")
def tile_and_matrix():
    m = make_counts(400, 120, density=0.1, seed=20)
    return Tile(encode_tile(m, n_chunks=4, level=3)).decode(values=np.int32), m


def test_gather_matches_scipy(tile_and_matrix):
    dec, m = tile_and_matrix
    rng = np.random.default_rng(0)
    rows = rng.permutation(400)[:150]
    for dtype in (np.float32, np.int32):
        b = dec.gather(rows, values_dtype=dtype)
        assert b.values.dtype == dtype and b.shape == (150, 120)
        assert same(b.to_scipy(), m[rows])


def test_gather_repeats_and_order(tile_and_matrix):
    dec, m = tile_and_matrix
    rows = [5, 5, 399, 0, 5, 17]
    assert same(dec.gather(rows).to_scipy(), m[rows])


def test_gather_empty(tile_and_matrix):
    dec, _ = tile_and_matrix
    b = dec.gather([])
    assert b.n_rows == 0 and b.nnz == 0 and list(b.indptr) == [0]


@pytest.mark.parametrize("rows", [[-1], [400], [0, 1000]])
def test_gather_out_of_range(tile_and_matrix, rows):
    dec, _ = tile_and_matrix
    with pytest.raises(IndexError):
        dec.gather(rows)


def test_gather_into_caller_buffers(tile_and_matrix):
    dec, m = tile_and_matrix
    rows = np.arange(0, 400, 3)
    need = int(row_offsets(dec.indptr, rows)[-1])
    oi, ov = np.full(need + 10, -1, np.int32), np.full(need + 10, -1, np.float32)
    b = dec.gather(rows, out_indices=oi, out_values=ov)
    assert same(b.to_scipy(), m[rows])
    assert np.shares_memory(b.indices, oi) and np.shares_memory(b.values, ov)
    assert (oi[need:] == -1).all()
    with pytest.raises(ValueError):
        dec.gather(rows, out_indices=oi[: need - 1], out_values=ov)


def test_float_tile_cannot_gather_as_int():
    m = make_counts(50, 30, seed=21)
    dec = Tile(encode_tile(m, level=3)).decode(values=np.float32)
    assert same(dec.gather([3, 1]).to_scipy(), m[[3, 1]])
    with pytest.raises(ValueError):
        dec.gather([3, 1], values_dtype=np.int32)


def test_to_scipy_roundtrip(tile_and_matrix):
    dec, m = tile_and_matrix
    assert same(dec.to_scipy(), m)
