# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os
import pickle
import time
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
from conftest import make_counts, same

from deltacells import (
    DatasetWriter,
    DeltaCellsDataset,
    FormatError,
    LocalBackend,
    Manifest,
    ThrottledBackend,
    open_dataset,
)

N_CELLS, N_GENES, TILE = 350, 90, 100


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """(root, original matrix in source gene order, gene_order, var_names)."""
    root = str(tmp_path_factory.mktemp("ds") / "ds")
    full = make_counts(N_CELLS, N_GENES, density=0.15, seed=30)
    order = np.random.default_rng(0).permutation(N_GENES)
    names = [f"g{j}" for j in range(N_GENES)]
    with DatasetWriter(
        root,
        n_genes=N_GENES,
        tile_size=TILE,
        gene_order=order,
        var_names=names,
        n_chunks=3,
        level=3,
        threads=2,
        metadata={"a": 1},
    ) as w:
        for lo in range(0, N_CELLS, TILE):
            w.add_tile(full[lo : lo + TILE])
    return root, full, order, names


@pytest.fixture
def expected(built):
    return built[1][:, built[2]]  # what the stored matrix should equal: columns permuted by gene_order


def test_manifest_and_files(built):
    root, _, order, names = built
    ds = open_dataset(root)
    m = ds.manifest
    assert (ds.n_cells, ds.n_genes, ds.n_tiles, ds.tile_size, len(ds)) == (N_CELLS, N_GENES, 4, TILE, N_CELLS)
    assert m.tile_cells == [100, 100, 100, 50] and list(ds.limits) == [100, 200, 300, 350]
    assert m.metadata == {"a": 1} and m.n_chunks == 3 and m.zstd_level == 3
    assert ds.tile_bounds(3) == (300, 350)
    assert list(ds.tile_of([0, 99, 100, 349])) == [0, 0, 1, 3]
    assert os.path.exists(os.path.join(root, "tiles", "tile_000003.dct"))
    assert np.array_equal(ds.gene_order, order)
    assert ds.var_names == [names[j] for j in order]
    assert m.tile_bytes == [os.path.getsize(os.path.join(root, Manifest.tile_name(i))) for i in range(4)]


def test_whole_dataset_roundtrip(built, expected):
    ds = open_dataset(built[0])
    assert same(ds.get_batch(np.arange(N_CELLS)).to_scipy(), expected)
    assert same(ds[:].to_scipy(), expected)


def test_random_batches_match_scipy(built, expected):
    ds = open_dataset(built[0], max_cached_tiles=2)
    rng = np.random.default_rng(1)
    for _ in range(25):
        idx = rng.integers(0, N_CELLS, size=int(rng.integers(1, 200)))  # unsorted, repeats, spans tiles
        for dtype in (np.float32, np.int32):
            b = ds.get_batch(idx, values_dtype=dtype)
            assert b.values.dtype == dtype
            assert same(b.to_scipy(), expected[idx])


def test_batch_across_tile_boundary(built, expected):
    ds = open_dataset(built[0], max_cached_tiles=1)
    idx = np.arange(95, 305)
    assert same(ds.get_batch(idx).to_scipy(), expected[idx])


def test_getitem_variants(built, expected):
    ds = open_dataset(built[0])
    assert same(ds[7].to_scipy(), expected[[7]])
    assert same(ds[-1].to_scipy(), expected[[N_CELLS - 1]])
    assert same(ds[10:40:3].to_scipy(), expected[10:40:3])
    assert same(ds[[3, 250, 3]].to_scipy(), expected[[3, 250, 3]])
    assert same(ds[np.array([349, 0])].to_scipy(), expected[[349, 0]])


def test_empty_and_invalid_indices(built):
    ds = open_dataset(built[0])
    b = ds.get_batch([])
    assert b.n_rows == 0 and list(b.indptr) == [0]
    for bad in ([-1], [N_CELLS], [0, 10**6]):
        with pytest.raises(IndexError):
            ds.get_batch(bad)
    with pytest.raises(IndexError):
        ds.prefetch([99])
    with pytest.raises(IndexError):
        ds.get_tile(99)


def test_out_buffers(built, expected):
    ds = open_dataset(built[0])
    idx = np.arange(20, 160)
    need = int(expected[idx].nnz)
    oi, ov = np.zeros(need + 5, np.int32), np.zeros(need + 5, np.float32)
    b = ds.get_batch(idx, out_indices=oi, out_values=ov)
    assert same(b.to_scipy(), expected[idx]) and np.shares_memory(b.values, ov)
    with pytest.raises(ValueError):
        ds.get_batch(idx, out_indices=oi[: need - 1], out_values=ov)


def test_decoded_cache_and_buffer_reuse(built, expected):
    ds = open_dataset(built[0], max_cached_tiles=2)
    seen = set()
    for t in [0, 1, 0, 2, 3, 1, 0, 2, 3, 3, 3]:
        lo, hi = ds.tile_bounds(t)
        assert same(ds.get_batch(np.arange(lo, hi)).to_scipy(), expected[lo:hi])
        seen.update(id(b[0]) for _, b in ds._decoded.values())
    assert len(ds._decoded) == 2
    assert len(seen) <= 3, "decode buffers should be recycled, not reallocated for every tile"
    assert ds.stats["decoded_cache_hits"] >= 1 and ds.stats["tiles_decoded"] < 11


def test_parallel_decode_threads(built, expected):
    ds = open_dataset(built[0], decode_threads=3)
    assert same(ds.get_batch(np.arange(N_CELLS)).to_scipy(), expected)
    ds.close()


def test_prefetch_overlaps_fetches(built, expected):
    slow = ThrottledBackend(LocalBackend(built[0]), latency_s=0.3)
    ds = DeltaCellsDataset(slow, io_threads=4, max_prefetch_tiles=8)  # manifest read also delayed; fine
    t0 = time.perf_counter()
    assert ds.prefetch([0, 1, 2, 3]) == 4 and ds.prefetch([0, 1]) == 0  # idempotent
    assert time.perf_counter() - t0 < 0.15, "prefetch must not block"
    time.sleep(0.6)  # all four fetches (in parallel) are done by now
    t0 = time.perf_counter()
    assert same(ds.get_batch(np.arange(N_CELLS)).to_scipy(), expected)
    assert time.perf_counter() - t0 < 0.25, "prefetched tiles should be ready"
    assert ds.stats["tiles_fetched"] == 4
    ds.close()


def test_no_prefetch_stalls_on_fetch(built):
    slow = ThrottledBackend(LocalBackend(built[0]), latency_s=0.25)
    ds = DeltaCellsDataset(slow)
    t0 = time.perf_counter()
    ds.get_batch([0])
    assert time.perf_counter() - t0 >= 0.25
    assert ds.stats["stall_seconds"] >= 0.2
    ds.close()


def test_prefetch_cells_and_cache_bound(built):
    ds = open_dataset(built[0], max_prefetch_tiles=2, io_threads=2)
    assert ds.prefetch_cells([0, 150, 250, 349]) >= 1
    for i in range(10):
        ds.prefetch([i % 4])
    time.sleep(0.2)
    ds.prefetch([0, 1, 2, 3])
    assert len(ds._tiles) <= 4
    ds.close()


def test_batch_needing_more_tiles_than_cache(built, expected):
    ds = open_dataset(built[0], max_prefetch_tiles=1, max_cached_tiles=1)
    idx = np.arange(N_CELLS)[::-1]
    assert same(ds.get_batch(idx).to_scipy(), expected[idx])


def test_pickle_roundtrip_and_independent_runtime(built, expected):
    ds = open_dataset(built[0], max_cached_tiles=1, decode_threads=2)
    ds.get_batch([1, 2, 3])
    clone = pickle.loads(pickle.dumps(ds))
    assert (
        clone.max_cached_tiles == 1
        and clone.decode_threads == 2
        and len(clone._decoded) == 0
        and clone.stats["tiles_fetched"] == 0
    )
    assert same(clone.get_batch([5, 205, 305]).to_scipy(), expected[[5, 205, 305]])


def test_context_manager_and_close(built):
    with open_dataset(built[0]) as ds:
        ds.get_batch([0])
    ds.close()  # idempotent
    assert ds._io_pool is None and len(ds._decoded) == 0
    assert ds.get_batch([0]).n_rows == 1  # usable again after close


def test_corrupt_tile_detected_with_verify(built, tmp_path):
    import shutil

    root = str(tmp_path / "c")
    shutil.copytree(built[0], root)
    p = os.path.join(root, "tiles", "tile_000001.dct")
    data = bytearray(Path(p).read_bytes())
    data[-5] ^= 0xFF
    Path(p).write_bytes(data)
    ds = open_dataset(root, verify=True)
    ds.get_batch([0])  # tile 0 is fine
    with pytest.raises(FormatError, match="CRC"):
        ds.get_batch([150])
    ds2 = open_dataset(root, verify=True)
    with pytest.raises(FormatError):
        ds2.get_batch([150])  # error is not cached as a result: the failed fetch is retried (and fails again)


def test_tile_not_matching_manifest(built, tmp_path):
    import shutil

    root = str(tmp_path / "m")
    shutil.copytree(built[0], root)
    shutil.copy(
        os.path.join(root, "tiles", "tile_000003.dct"), os.path.join(root, "tiles", "tile_000000.dct")
    )  # 50 cells, not 100
    with pytest.raises(ValueError, match="manifest"):
        open_dataset(root).get_batch([0])


def test_missing_manifest(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_dataset(str(tmp_path))


def test_invalid_options(built):
    for kw in ({"max_cached_tiles": 0}, {"io_threads": 0}, {"decode_threads": 0}, {"max_prefetch_tiles": 0}):
        with pytest.raises(ValueError):
            open_dataset(built[0], **kw)


# ---------------------------------------------------------------------------------------------- writer


def test_writer_refuses_nonempty_dir_without_overwrite(tmp_path):
    (tmp_path / "x").write_text("hi")
    with pytest.raises(FileExistsError):
        DatasetWriter(str(tmp_path), n_genes=5, tile_size=10)
    w = DatasetWriter(str(tmp_path), n_genes=5, tile_size=10, overwrite=True)
    assert not (tmp_path / "x").exists()
    w.add_tile(make_counts(10, 5))
    w.close()


def test_writer_tile_rules(tmp_path):
    w = DatasetWriter(str(tmp_path / "w"), n_genes=20, tile_size=10, level=3)
    with pytest.raises(ValueError, match="genes"):
        w.add_tile(make_counts(10, 21))
    with pytest.raises(ValueError, match="between 1 and"):
        w.add_tile(make_counts(11, 20))
    with pytest.raises(ValueError, match="between 1 and"):
        w.add_tile(sp.csr_matrix((0, 20)))
    w.add_tile(make_counts(10, 20))
    w.add_tile(make_counts(4, 20))  # short last tile
    with pytest.raises(ValueError, match="last tile"):
        w.add_tile(make_counts(10, 20))
    m = w.close()
    assert m.tile_cells == [10, 4]
    with pytest.raises(RuntimeError):
        w.add_tile(make_counts(1, 20))
    with pytest.raises(RuntimeError):
        w.close()


def test_writer_without_tiles_or_after_exception_leaves_no_manifest(tmp_path):
    w = DatasetWriter(str(tmp_path / "a"), n_genes=5, tile_size=10)
    with pytest.raises(ValueError, match="no tiles"):
        w.close()
    with pytest.raises(RuntimeError, match="boom"):
        with DatasetWriter(str(tmp_path / "b"), n_genes=5, tile_size=10, level=3) as w2:
            w2.add_tile(make_counts(10, 5))
            raise RuntimeError("boom")
    assert not os.path.exists(tmp_path / "b" / "manifest.json")
    with pytest.raises(FileNotFoundError):
        open_dataset(str(tmp_path / "b"))


def test_writer_argument_validation(tmp_path):
    with pytest.raises(ValueError):
        DatasetWriter(str(tmp_path / "x"), n_genes=5, tile_size=0)
    with pytest.raises(ValueError):
        DatasetWriter(str(tmp_path / "y"), n_genes=5, tile_size=5, gene_order=[0, 1, 1, 3, 4])
    with pytest.raises(ValueError):
        DatasetWriter(str(tmp_path / "z"), n_genes=5, tile_size=5, var_names=["a"])


def test_dataset_without_gene_order_or_names(tmp_path):
    m = make_counts(30, 12, seed=31)
    with DatasetWriter(str(tmp_path / "p"), n_genes=12, tile_size=10, level=3) as w:
        for lo in range(0, 30, 10):
            w.add_tile(m[lo : lo + 10])
    ds = open_dataset(str(tmp_path / "p"))
    assert ds.gene_order is None and ds.var_names is None
    assert same(ds[:].to_scipy(), m)


def test_manifest_validation():
    good = Manifest(
        n_cells=30,
        n_genes=5,
        tile_size=10,
        tile_cells=[10, 10, 10],
        tile_nnz=[1, 2, 3],
        tile_bytes=[5, 5, 5],
        n_chunks=1,
        zstd_level=3,
    )
    assert Manifest.from_json(good.to_json()) == good
    import json

    d = json.loads(good.to_json())
    for mutate in (
        lambda d: d.update(format="other"),
        lambda d: d.update(version=99),
        lambda d: d.update(n_cells=31),
        lambda d: d.update(tile_cells=[10, 5, 15], n_cells=30),
        lambda d: d.update(tile_nnz=[1, 2]),
    ):
        bad = json.loads(json.dumps(d))
        mutate(bad)
        with pytest.raises(ValueError):
            Manifest.from_json(json.dumps(bad))


def test_batch_nnz_matches_get_batch(built, expected):
    ds = open_dataset(built[0])
    rng = np.random.default_rng(5)
    for idx in (rng.integers(0, N_CELLS, 77), np.arange(90, 210), np.array([], dtype=np.int64)):
        assert ds.batch_nnz(idx) == int(expected[idx].nnz) == ds.get_batch(idx).nnz
    fresh = open_dataset(built[0])
    fresh.batch_nnz([5, 150])
    assert fresh.stats["tiles_decoded"] == 0 and fresh.stats["tiles_fetched"] == 2


def test_concurrent_get_batch_from_threads(built, expected):
    from concurrent.futures import ThreadPoolExecutor

    ds = open_dataset(built[0], max_cached_tiles=1)
    rng = np.random.default_rng(6)
    batches = [rng.integers(0, N_CELLS, 40) for _ in range(40)]
    with ThreadPoolExecutor(6) as pool:
        results = list(pool.map(lambda idx: ds.get_batch(idx).to_scipy(), batches))
    assert all(same(r, expected[idx]) for r, idx in zip(results, batches))


def test_add_tile_does_not_modify_the_callers_matrix(tmp_path):
    m = make_counts(40, 30, density=0.3, seed=32)
    snap = (m.data.copy(), m.indices.copy(), m.indptr.copy())
    order = np.random.default_rng(1).permutation(30)
    with DatasetWriter(str(tmp_path / "d"), n_genes=30, tile_size=40, gene_order=order, level=3) as w:
        w.add_tile(m)
    assert np.array_equal(m.data, snap[0]) and np.array_equal(m.indices, snap[1]) and np.array_equal(m.indptr, snap[2])
    assert same(open_dataset(str(tmp_path / "d"))[:].to_scipy(), m[:, order])
