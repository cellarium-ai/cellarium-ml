# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import scipy.sparse as sp
from conftest import make_counts, same

from deltacells import Tile, encode_tile
from deltacells.format import MAX_GENES, MAX_VALUE, parse_tile_info


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (5, 1), (100, 50), (500, 300)])
@pytest.mark.parametrize("density", [0.0, 0.02, 0.3, 1.0])
@pytest.mark.parametrize("n_chunks", [1, 3, 8, 1000])
def test_roundtrip(shape, density, n_chunks):
    m = make_counts(*shape, density=density, seed=hash((shape, density)) % 1000)
    t = Tile(encode_tile(m, n_chunks=n_chunks, level=3))
    assert t.n_cells == shape[0] and t.n_genes == shape[1] and t.nnz == m.nnz
    assert 1 <= t.n_chunks <= min(n_chunks, shape[0])
    for dtype in (np.int32, np.float32):
        d = t.decode(values=dtype)
        assert d.values.dtype == dtype and d.indices.dtype == np.int32
        assert same(d.to_scipy(), m)


@pytest.mark.parametrize("level", [1, 3, 12, 19])
def test_levels(level):
    m = make_counts(300, 200, seed=3)
    assert same(Tile(encode_tile(m, level=level)).to_scipy(), m)


def test_higher_levels_not_larger():
    m = make_counts(2000, 500, density=0.1, seed=4)
    sizes = [len(encode_tile(m, level=lv, n_chunks=1)) for lv in (1, 19)]
    assert sizes[1] <= sizes[0]


def test_extreme_values_and_genes():
    n_genes = MAX_GENES
    rows, cols, vals = [0, 0, 0, 1, 2], [0, 1, n_genes - 1, n_genes - 1, 12345], [1, MAX_VALUE, MAX_VALUE, 1, 777]
    m = sp.csr_matrix((vals, (rows, cols)), shape=(3, n_genes), dtype=np.float32)
    d = Tile(encode_tile(m, level=3)).decode(values=np.int32)
    assert same(d.to_scipy(), m)
    assert d.values.max() == MAX_VALUE and d.indices.max() == n_genes - 1


def test_empty_cells_interspersed():
    m = make_counts(60, 40, density=0.05, seed=5).tolil()
    for i in (0, 1, 7, 30, 59):
        m.rows[i], m.data[i] = [], []
    m = m.tocsr()
    t = Tile(encode_tile(m, n_chunks=5, level=3))
    assert same(t.to_scipy(), m)
    assert (np.diff(t.indptr) == 0).sum() >= 5


def test_all_empty_tile():
    m = sp.csr_matrix((20, 30), dtype=np.float32)
    t = Tile(encode_tile(m))
    assert t.nnz == 0 and same(t.to_scipy(), m)


def test_zero_cell_tile():
    m = sp.csr_matrix((0, 30), dtype=np.float32)
    t = Tile(encode_tile(m))
    assert t.n_cells == 0 and t.n_chunks == 0
    d = t.decode()
    assert d.nnz == 0 and len(d.indices) == 0


def test_dense_and_other_formats_accepted():
    m = make_counts(30, 20, density=0.3, seed=6)
    for src in (m.toarray(), m.tocsc(), m.tocoo(), m.astype(np.int64)):
        assert same(Tile(encode_tile(src, level=3)).to_scipy(), m)


def test_duplicates_summed_unsorted_and_explicit_zeros_dropped():
    rows = [0, 0, 0, 1, 1]
    cols = [5, 5, 2, 3, 4]
    vals = [1, 2, 4, 0, 6]  # (0,5) duplicated; (1,3) explicit zero
    m = sp.coo_matrix((vals, (rows, cols)), shape=(2, 10)).tocsr()
    expected = sp.csr_matrix(
        np.array([[0, 0, 4, 0, 0, 3, 0, 0, 0, 0], [0, 0, 0, 0, 6, 0, 0, 0, 0, 0]], dtype=np.float32)
    )
    raw = sp.csr_matrix(
        (np.array([1, 2, 4, 0, 6], dtype=np.float32), np.array([5, 5, 2, 3, 4]), np.array([0, 3, 5])), shape=(2, 10)
    )
    t = Tile(encode_tile(raw, level=3))
    assert same(t.to_scipy(), expected) and t.nnz == 3
    assert same(Tile(encode_tile(m, level=3)).to_scipy(), m)


@pytest.mark.parametrize(
    "data,msg",
    [
        ([-1.0], "values must lie"),
        ([65536.0], "values must lie"),
        ([1.5], "integer"),
        ([np.nan], "integer"),
        ([np.inf], "integer"),
    ],
)
def test_invalid_values_rejected(data, msg):
    m = sp.csr_matrix((np.array(data, dtype=np.float32), ([0], [0])), shape=(1, 3))
    with pytest.raises(ValueError, match=msg):
        encode_tile(m)


def test_too_many_genes_rejected():
    with pytest.raises(ValueError, match="n_genes"):
        encode_tile(sp.csr_matrix((2, MAX_GENES + 1), dtype=np.float32))


def test_invalid_arguments():
    m = make_counts(10, 10)
    with pytest.raises(ValueError):
        encode_tile(m, n_chunks=0)
    with pytest.raises(ValueError):
        encode_tile(m, level=100)


def test_deterministic_across_thread_counts():
    m = make_counts(400, 100, seed=7)
    a = encode_tile(m, n_chunks=6, level=5, threads=1)
    assert a == encode_tile(m, n_chunks=6, level=5, threads=4) == encode_tile(m, n_chunks=6, level=5, threads=1)


def test_chunks_are_balanced_by_nonzeros():
    m = make_counts(1000, 200, density=0.1, seed=8)
    info = parse_tile_info(encode_tile(m, n_chunks=8, level=3))
    sizes = [c.nnz for c in info.chunks]
    assert len(sizes) == 8 and max(sizes) < 1.5 * m.nnz / 8


@pytest.mark.parametrize("threads", [1, 2, 4, 16])
def test_parallel_decode_matches_serial(threads):
    m = make_counts(800, 300, seed=9)
    t = Tile(encode_tile(m, n_chunks=8, level=3))
    assert same(t.decode(threads=threads).to_scipy(), m)


def test_preallocated_buffers_and_reuse():
    m = make_counts(300, 100, seed=10)
    t = Tile(encode_tile(m, n_chunks=4, level=3))
    idx, val = np.full(t.nnz + 50, -7, dtype=np.int32), np.full(t.nnz + 50, -7, dtype=np.float32)
    scratch = np.empty(2 * t.info.scratch_nbytes, dtype=np.uint8)
    for _ in range(2):  # reuse
        d = t.decode(values=np.float32, threads=2, out_indices=idx, out_values=val, scratch=scratch)
        assert same(d.to_scipy(), m)
    assert (idx[t.nnz :] == -7).all() and (val[t.nnz :] == -7).all()  # nothing written past nnz


def test_wrong_buffers_rejected():
    t = Tile(encode_tile(make_counts(100, 50, seed=11), level=3))
    ok_i, ok_v = np.empty(t.nnz, np.int32), np.empty(t.nnz, np.int32)
    with pytest.raises(ValueError):
        t.decode(out_indices=np.empty(t.nnz - 1, np.int32), out_values=ok_v)
    with pytest.raises(ValueError):
        t.decode(out_indices=np.empty(t.nnz, np.int64), out_values=ok_v)
    with pytest.raises(ValueError):
        t.decode(out_indices=ok_i, out_values=np.empty(t.nnz, np.float32))  # dtype differs from values=int32
    with pytest.raises(ValueError):
        t.decode(out_indices=ok_i, out_values=ok_v, scratch=np.empty(1, np.uint8))
    with pytest.raises(ValueError):
        t.decode(values=np.int64)
    ro = np.empty(t.nnz, np.int32)
    ro.setflags(write=False)
    with pytest.raises(ValueError):
        t.decode(out_indices=ro, out_values=ok_v)


def test_accepts_various_buffer_types(tmp_path):
    import mmap

    blob = encode_tile(make_counts(50, 40, seed=12), level=3)
    m = Tile(blob).to_scipy()
    p = tmp_path / "t.dct"
    p.write_bytes(blob)
    with open(p, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        assert same(Tile(mm).to_scipy(), m)
    assert same(Tile(bytearray(blob)).to_scipy(), m)
    assert same(Tile(memoryview(blob)).to_scipy(), m)
    assert same(Tile(np.frombuffer(blob, dtype=np.uint8)).to_scipy(), m)


def test_misaligned_blob_is_fine():
    blob = encode_tile(make_counts(50, 40, seed=13), level=3)
    shifted = np.empty(len(blob) + 3, dtype=np.uint8)
    shifted[3:] = np.frombuffer(blob, dtype=np.uint8)
    t = Tile(shifted[3:])  # data pointer is not 8-byte aligned
    assert same(t.to_scipy(), Tile(blob).to_scipy())


def test_fuzz_roundtrip_and_gather():
    rng = np.random.default_rng(1234)
    for _ in range(60):
        n_cells, n_genes = int(rng.integers(1, 120)), int(rng.integers(1, 200))
        m = make_counts(
            n_cells,
            n_genes,
            density=float(rng.choice([0.0, 0.01, 0.1, 0.6, 1.0])),
            seed=int(rng.integers(1 << 30)),
            max_value=int(rng.choice([1, 3, 255, 65535])),
        )
        t = Tile(
            encode_tile(
                m, n_chunks=int(rng.integers(1, 12)), level=int(rng.choice([1, 3, 9])), threads=int(rng.integers(1, 4))
            )
        )
        d = t.decode(values=np.int32, threads=int(rng.integers(1, 5)))
        assert same(d.to_scipy(), m)
        rows = rng.integers(0, n_cells, size=int(rng.integers(0, 2 * n_cells + 1)))
        assert same(
            d.gather(rows, values_dtype=np.float32).to_scipy(), m[rows] if len(rows) else sp.csr_matrix((0, n_genes))
        )


def test_permute_columns_does_not_modify_its_input():
    from deltacells import permute_columns

    m = make_counts(120, 60, density=0.3, seed=80)
    snapshot = (m.data.copy(), m.indices.copy(), m.indptr.copy())
    order = np.random.default_rng(0).permutation(60)
    out = permute_columns(m, order)
    assert same(out, m[:, order])
    assert (
        np.array_equal(m.data, snapshot[0])
        and np.array_equal(m.indices, snapshot[1])
        and np.array_equal(m.indptr, snapshot[2])
    )
    assert not np.shares_memory(out.data, m.data) and not np.shares_memory(out.indptr, m.indptr)


def test_encode_tile_does_not_modify_its_input():
    m = make_counts(100, 50, density=0.2, seed=81)
    unsorted = sp.csr_matrix(
        (m.data[::-1].copy(), m.indices[::-1].copy(), m.indptr.copy()), shape=m.shape
    )  # not canonical
    snaps = [x.copy() for x in (m.data, m.indices, m.indptr)]
    encode_tile(m, level=3)
    uns = [x.copy() for x in (unsorted.data, unsorted.indices, unsorted.indptr)]
    encode_tile(unsorted, level=3)
    assert all(np.array_equal(a, b) for a, b in zip(snaps, (m.data, m.indices, m.indptr)))
    assert all(np.array_equal(a, b) for a, b in zip(uns, (unsorted.data, unsorted.indices, unsorted.indptr)))
