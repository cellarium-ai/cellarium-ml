# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from conftest import make_counts, same

anndata = pytest.importorskip("anndata")

from deltacells import open_dataset  # noqa: E402
from deltacells.cli import main  # noqa: E402
from deltacells.convert import convert_h5ad, resolve_files  # noqa: E402

N, G = 530, 40


def write_shards(directory, sizes, full, var_names=None, prefix="s"):
    os.makedirs(directory, exist_ok=True)
    names = var_names or [f"gene{j}" for j in range(full.shape[1])]
    lo = 0
    for i, n in enumerate(sizes):
        obs = pd.DataFrame(index=[f"cell{j}" for j in range(lo, lo + n)])
        anndata.AnnData(X=full[lo : lo + n].astype(np.float32), obs=obs, var=pd.DataFrame(index=names)).write_h5ad(
            f"{directory}/{prefix}_{i:03d}.h5ad"
        )
        lo += n
    assert lo == full.shape[0]


@pytest.fixture(scope="module")
def full():
    return make_counts(N, G, density=0.2, seed=40, max_value=30)


def test_natural_sort_and_resolve(tmp_path):
    for n in (2, 10, 1):
        (tmp_path / f"s_{n}.h5ad").write_bytes(b"")
    assert [os.path.basename(f) for f in resolve_files([str(tmp_path / "s_*.h5ad")])] == [
        "s_1.h5ad",
        "s_2.h5ad",
        "s_10.h5ad",
    ]
    with pytest.raises(FileNotFoundError):
        resolve_files([str(tmp_path / "nothing*")])


def test_resolve_keeps_the_given_order_without_sorting(tmp_path):
    paths = [str(tmp_path / f"s_{n}.h5ad") for n in (10, 2, 1)]
    for p in paths:
        open(p, "wb").close()
    assert resolve_files(paths, sort=False) == paths
    assert resolve_files([*paths, paths[0]], sort=False) == paths  # duplicates are dropped
    assert resolve_files(paths) == [paths[2], paths[1], paths[0]]
    with pytest.raises(FileNotFoundError):  # a missing file is an error, not silently skipped
        resolve_files([*paths, str(tmp_path / "missing.h5ad")], sort=False)


def test_jagged_and_perfect_shards_give_identical_datasets(tmp_path, full):
    write_shards(tmp_path / "perfect", [100, 100, 100, 100, 100, 30], full)
    write_shards(tmp_path / "jagged", [37, 250, 3, 100, 140], full)
    kw = dict(tile_size=100, sort_genes=False, n_chunks=3, level=3, threads=1, log=None)
    ma = convert_h5ad(str(tmp_path / "perfect" / "*.h5ad"), str(tmp_path / "A"), **kw)
    mb = convert_h5ad(str(tmp_path / "jagged" / "*.h5ad"), str(tmp_path / "B"), **kw)
    assert ma.tile_cells == mb.tile_cells == [100] * 5 + [30]
    for i in range(ma.n_tiles):
        a = Path(tmp_path / "A" / "tiles" / f"tile_{i:06d}.dct").read_bytes()
        b = Path(tmp_path / "B" / "tiles" / f"tile_{i:06d}.dct").read_bytes()
        assert a == b, f"tile {i} differs"
    assert same(open_dataset(str(tmp_path / "B"))[:].to_scipy(), full)


def test_sort_genes_orders_by_the_totals_of_all_files(tmp_path, full):
    write_shards(tmp_path / "in", [200, 200, 130], full)
    out = str(tmp_path / "out")
    convert_h5ad(str(tmp_path / "in" / "*.h5ad"), out, tile_size=150, n_chunks=2, level=3, threads=1, log=None)
    ds = open_dataset(out)
    totals = np.asarray(full.sum(0)).ravel()
    order = ds.gene_order
    assert sorted(order) == list(range(G)) and (np.diff(totals[order]) <= 0).all()
    assert same(ds[:].to_scipy(), full[:, order])
    assert ds.var_names == [f"gene{j}" for j in order]
    meta = ds.manifest.metadata
    assert meta["sort_genes"] is True and meta["n_source_files"] == 3 and meta["sort_genes_n_files"] == 3


def test_sort_genes_max_files_samples_files_evenly(tmp_path, full):
    write_shards(tmp_path / "in", [100, 100, 100, 100, 100, 30], full)
    out = str(tmp_path / "out")
    kw = dict(tile_size=100, n_chunks=2, level=3, threads=1, workers=1, log=None)
    convert_h5ad(str(tmp_path / "in" / "*.h5ad"), out, sort_genes_max_files=2, **kw)
    ds = open_dataset(out)
    sampled = np.asarray(full[:100].sum(0) + full[500:].sum(0)).ravel()  # the first and the last file
    assert ds.manifest.metadata["sort_genes_n_files"] == 2
    assert (np.diff(sampled[ds.gene_order]) <= 0).all()
    assert same(ds[:].to_scipy(), full[:, ds.gene_order])
    with pytest.raises(ValueError, match="sort_genes_max_files"):
        convert_h5ad(str(tmp_path / "in" / "*.h5ad"), out, sort_genes_max_files=0, overwrite=True, **kw)


def test_workers_give_the_same_dataset_as_one_process(tmp_path, full):
    write_shards(tmp_path / "in", [37, 250, 3, 100, 140], full)  # tiles that span several files
    kw = dict(tile_size=64, n_chunks=3, level=3, threads=1, log=None)
    one = convert_h5ad(str(tmp_path / "in" / "*.h5ad"), str(tmp_path / "A"), workers=1, **kw)
    many = convert_h5ad(str(tmp_path / "in" / "*.h5ad"), str(tmp_path / "B"), workers=3, **kw)
    assert one.tile_cells == many.tile_cells == [64] * 8 + [18]
    assert one.tile_bytes == many.tile_bytes and one.tile_nnz == many.tile_nnz
    for i in range(one.n_tiles):
        a = Path(tmp_path / "A" / "tiles" / f"tile_{i:06d}.dct").read_bytes()
        b = Path(tmp_path / "B" / "tiles" / f"tile_{i:06d}.dct").read_bytes()
        assert a == b, f"tile {i} differs"
    ds = open_dataset(str(tmp_path / "B"))
    assert same(ds[:].to_scipy(), full[:, ds.gene_order])


def test_scanning_the_files_in_processes_gives_the_same_dataset(tmp_path, full):
    write_shards(tmp_path / "in", [100, 100, 100, 100, 100, 30], full)
    kw = dict(tile_size=100, level=3, threads=1, workers=1, log=None)
    threaded = convert_h5ad(str(tmp_path / "in" / "*.h5ad"), str(tmp_path / "A"), scan_workers=1, **kw)
    in_processes = convert_h5ad(str(tmp_path / "in" / "*.h5ad"), str(tmp_path / "B"), scan_workers=3, **kw)
    assert threaded.to_json() == in_processes.to_json()  # also the obs fingerprint: the schema is the same
    for name in (
        "tiles/tile_000000.dct",
        "tiles/tile_000005.dct",
        "obs/obs_0000.parquet",
        "var.parquet",
        "gene_order.npy",
    ):
        assert (tmp_path / "A" / name).read_bytes() == (tmp_path / "B" / name).read_bytes(), name


def test_which_files_are_scanned_in_processes():
    from deltacells.convert import _SCAN_MIN_FILES, _scan_processes

    assert _scan_processes(10_000, None, 1) == 0  # asked for this process
    assert _scan_processes(_SCAN_MIN_FILES - 1, None, None) == 0  # too few files to be worth starting processes
    assert 2 <= _scan_processes(10_000, None, None) <= 24
    assert (
        _scan_processes(10_000, object(), None) == 0
    )  # a filesystem that is passed in cannot be shared with processes
    assert _scan_processes(5, None, 8) == 5  # never more processes than files
    with pytest.raises(ValueError, match="filesystem"):
        _scan_processes(100, object(), 4)
    with pytest.raises(ValueError, match="at least 1"):
        _scan_processes(100, None, 0)


def test_a_scan_process_reports_an_unreadable_file(tmp_path, full):
    write_shards(tmp_path / "in", [300, 230], full)
    (tmp_path / "in" / "s_002.h5ad").write_bytes(b"this is not an hdf5 file")
    with pytest.raises(OSError):
        convert_h5ad(str(tmp_path / "in" / "*.h5ad"), str(tmp_path / "out"), scan_workers=2, log=None)
    assert not os.path.exists(tmp_path / "out")


def test_progress_lines_name_the_phases_and_nothing_keeps_running_afterwards(tmp_path, full):
    import threading

    write_shards(tmp_path / "in", [300, 230], full)
    messages: list[str] = []
    kw = dict(tile_size=100, level=3, threads=1, workers=1, log=messages.append)
    convert_h5ad(str(tmp_path / "in" / "*.h5ad"), str(tmp_path / "out"), progress_interval=0.05, **kw)
    phases = [
        "reading the metadata of 2 files",
        "scan: 2 files in",
        "gene order: 2 files in",
        "tiles: 6 tiles in",
        "finishing: done in",
        "done in",
    ]
    positions = [next(i for i, m in enumerate(messages) if m.startswith(p)) for p in phases]
    assert positions == sorted(positions)  # in this order

    messages.clear()  # no periodic lines with an interval of 0: only the start and the end of each phase
    convert_h5ad(str(tmp_path / "in" / "*.h5ad"), str(tmp_path / "out"), overwrite=True, progress_interval=0, **kw)
    assert not [m for m in messages if "ETA" in m]

    write_shards(tmp_path / "bad", [10], full[:10], var_names=[f"other{j}" for j in range(G)], prefix="z")
    with pytest.raises(ValueError, match="var_names"):  # a failure stops the thread too
        convert_h5ad(
            [str(tmp_path / "in" / "s_000.h5ad"), str(tmp_path / "bad" / "z_000.h5ad")], str(tmp_path / "x"), **kw
        )
    assert not [t for t in threading.enumerate() if t.name == "deltacells-progress"]


def test_dense_x_gives_the_same_dataset_as_csr(tmp_path, full):
    os.makedirs(tmp_path / "dense")
    obs = pd.DataFrame(index=[f"cell{j}" for j in range(N)])
    var = pd.DataFrame(index=[f"gene{j}" for j in range(G)])
    anndata.AnnData(X=full.toarray().astype(np.float32), obs=obs, var=var).write_h5ad(tmp_path / "dense" / "x.h5ad")
    write_shards(tmp_path / "csr", [N], full)
    kw = dict(tile_size=100, n_chunks=2, level=3, threads=1, workers=2, log=None)
    convert_h5ad(str(tmp_path / "dense" / "*.h5ad"), str(tmp_path / "A"), **kw)
    convert_h5ad(str(tmp_path / "csr" / "*.h5ad"), str(tmp_path / "B"), **kw)
    for i in range(6):
        a = Path(tmp_path / "A" / "tiles" / f"tile_{i:06d}.dct").read_bytes()
        b = Path(tmp_path / "B" / "tiles" / f"tile_{i:06d}.dct").read_bytes()
        assert a == b, f"tile {i} differs"


def test_obs_follows_the_cells_across_files_and_workers(tmp_path, full):
    write_shards(tmp_path / "in", [37, 250, 3, 100, 140], full)
    out = str(tmp_path / "out")
    convert_h5ad(str(tmp_path / "in" / "*.h5ad"), out, tile_size=64, level=3, threads=1, workers=3, log=None)
    ds = open_dataset(out)
    assert list(ds.obs.to_pandas("obs_names")["obs_names"]) == [f"cell{j}" for j in range(N)]


def test_default_workers_are_limited_by_memory(monkeypatch):
    from deltacells import convert

    monkeypatch.setattr(convert, "_cpu_count", lambda: 16)
    monkeypatch.setattr(convert, "_available_memory", lambda: 10 * 2**30)
    assert convert._default_workers(100, 1 * 2**30) == 7  # 70% of 10 GiB
    assert convert._default_workers(100, 100 * 2**30) == 1  # never fewer than one
    assert convert._default_workers(3, 1) == 3  # not more than the jobs
    monkeypatch.setattr(convert, "_available_memory", lambda: None)
    assert convert._default_workers(100, 2**40) == 16


def test_tile_pieces_cover_the_cells_in_order():
    from deltacells.convert import _tile_pieces

    files, offsets = ["a", "empty", "b", "c"], [0, 5, 5, 17, 20]
    pieces = [p for i in range(3) for p in _tile_pieces(i, 8, offsets, files)]
    assert _tile_pieces(0, 8, offsets, files) == [("a", 0, 5), ("b", 0, 3)]
    assert _tile_pieces(1, 8, offsets, files) == [("b", 3, 11)]
    assert _tile_pieces(2, 8, offsets, files) == [("b", 11, 12), ("c", 0, 3)]
    assert sum(e - s for _, s, e in pieces) == 20


def test_rejects_mismatched_var_names(tmp_path, full):
    write_shards(tmp_path / "in", [300], full[:300], prefix="a")
    write_shards(tmp_path / "in", [230], full[300:], var_names=[f"other{j}" for j in range(G)], prefix="b")
    with pytest.raises(ValueError, match="var_names"):
        convert_h5ad(str(tmp_path / "in" / "*.h5ad"), str(tmp_path / "out"), tile_size=100, log=None)


@pytest.mark.parametrize("sort_genes", [False, True])
@pytest.mark.parametrize("workers", [1, 2])
def test_large_counts_and_more_than_65536_genes_survive_the_conversion(tmp_path, sort_genes, workers):
    g = 70_000
    rows = [[0, 66_500, 69_999], [69_000], [5, 6, 7, 120], [], [0, 33_000, 69_999], [1, 2]] * 5
    entries = [
        (r, c, v) for r, genes in enumerate(rows) for c, v in zip(genes, [3, 70_000, 5, 3_000_000][: len(genes)])
    ]
    r, c, v = zip(*entries)
    full = sp.csr_matrix((v, (r, c)), shape=(len(rows), g), dtype=np.int64)
    write_shards(tmp_path / "in", [12, 18], full)  # the float32 in the h5ad files holds these values exactly
    out = str(tmp_path / "out")
    kw = dict(tile_size=9, n_chunks=2, level=3, threads=1, log=None, sort_genes=sort_genes, workers=workers)
    manifest = convert_h5ad(str(tmp_path / "in" / "*.h5ad"), out, **kw)
    ds = open_dataset(out)
    order = ds.gene_order if sort_genes else np.arange(g)
    assert manifest.n_genes == g and same(ds[:].to_scipy(), full[:, order])


def test_a_tile_of_raw_counts_gone_wrong_is_refused_with_its_tile_and_files(tmp_path):
    n = 1100  # every nonzero of the tile is above 65535: not counts
    x = sp.csr_matrix(np.full((n, 2), 100_000, dtype=np.float32))
    os.makedirs(tmp_path / "in")
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n)])
    anndata.AnnData(X=x, obs=obs, var=pd.DataFrame(index=["g0", "g1"])).write_h5ad(tmp_path / "in" / "x.h5ad")
    with pytest.raises(ValueError, match=r"tile 0 \(from .*x\.h5ad\).*does not look like raw counts"):
        convert_h5ad(str(tmp_path / "in" / "*.h5ad"), str(tmp_path / "out"), tile_size=n, log=None)
    assert not os.path.exists(tmp_path / "out")


def test_rejects_non_integer_counts(tmp_path):
    os.makedirs(tmp_path / "in")
    x = sp.csr_matrix(np.array([[0.5, 0.0], [1.0, 2.0]], dtype=np.float32))
    anndata.AnnData(X=x, obs=pd.DataFrame(index=["a", "b"]), var=pd.DataFrame(index=["g0", "g1"])).write_h5ad(
        tmp_path / "in" / "x.h5ad"
    )
    with pytest.raises(ValueError, match="integer"):
        convert_h5ad(str(tmp_path / "in" / "*.h5ad"), str(tmp_path / "out"), tile_size=10, log=None)


def test_refuses_existing_output_without_overwrite(tmp_path, full):
    write_shards(tmp_path / "in", [530], full)
    out = str(tmp_path / "out")
    convert_h5ad(str(tmp_path / "in" / "*.h5ad"), out, tile_size=200, level=3, log=None)
    with pytest.raises(FileExistsError):
        convert_h5ad(str(tmp_path / "in" / "*.h5ad"), out, tile_size=200, level=3, log=None)
    convert_h5ad(str(tmp_path / "in" / "*.h5ad"), out, tile_size=250, level=3, log=None, overwrite=True)
    assert open_dataset(out).tile_size == 250


def test_cli_convert_from_a_file_list(tmp_path, full, monkeypatch):
    import io

    write_shards(tmp_path / "in", [100, 200, 230], full)
    shard = [str(tmp_path / "in" / f"s_{i:03d}.h5ad") for i in range(3)]
    listing = tmp_path / "files.txt"
    listing.write_text(f"# the shards, last first\n{shard[2]}\n\n  {shard[0]}  \n{shard[1]}\n")
    base = ["convert", "--no-sort-genes", "--level", "3", "--threads", "1", "--tile-size", "100"]

    out = str(tmp_path / "ordered")  # --keep-order: the cells follow the list
    assert main([*base, "--file-list", str(listing), "--keep-order", "--output", out]) == 0
    assert same(open_dataset(out)[:].to_scipy(), sp.vstack([full[300:], full[:100], full[100:300]]))

    out = str(tmp_path / "sorted")  # by default the files are naturally sorted
    assert main([*base, "--file-list", str(listing), "--output", out]) == 0
    assert same(open_dataset(out)[:].to_scipy(), full)

    out = str(tmp_path / "stdin")  # '-' reads the list from stdin, and it combines with --h5ad-glob
    monkeypatch.setattr("sys.stdin", io.StringIO(f"{shard[0]}\n{shard[1]}\n"))
    assert main([*base, "--file-list", "-", "--h5ad-glob", shard[2], "--output", out]) == 0
    assert same(open_dataset(out)[:].to_scipy(), full)

    assert main([*base, "--output", str(tmp_path / "none")]) == 2  # no files at all


def test_cli_passes_gs_paths_through_untouched(tmp_path, monkeypatch):
    seen = {}

    def fake_convert(paths, output, **kwargs):
        seen.update(paths=paths, output=output, **kwargs)

    monkeypatch.setattr("deltacells.convert.convert_h5ad", fake_convert)
    listing = tmp_path / "files.txt"
    listing.write_text("gs://bucket/in/a.h5ad\ngs://bucket/in/b[1].h5ad\n/data/c[2].h5ad\n")
    argv = ["convert", "--file-list", str(listing), "--output", "gs://bucket/out", "--staging-dir", "/scratch"]
    assert main([*argv, "--keep-order"]) == 0
    assert seen["paths"] == ["gs://bucket/in/a.h5ad", "gs://bucket/in/b[1].h5ad", "/data/c[[]2].h5ad"]  # local: literal
    assert seen["output"] == "gs://bucket/out" and seen["sort_files"] is False and seen["staging_dir"] == "/scratch"
    assert main(argv) == 0 and seen["sort_files"] is True
    assert seen["scan_workers"] is None and seen["progress_interval"] == 60.0  # the defaults
    assert main([*argv, "--scan-workers", "4", "--progress-interval", "5"]) == 0
    assert seen["scan_workers"] == 4 and seen["progress_interval"] == 5.0


def test_cli_convert_info_verify(tmp_path, full, capsys):
    write_shards(tmp_path / "in", [300, 230], full)
    out = str(tmp_path / "out")
    assert (
        main(
            [
                "convert",
                "--h5ad-glob",
                str(tmp_path / "in" / "*.h5ad"),
                "--output",
                out,
                "--tile-size",
                "200",
                "--level",
                "3",
                "--threads",
                "1",
            ]
        )
        == 0
    )
    assert main(["info", out]) == 0
    text = capsys.readouterr().out
    assert "530 x 40" in text and "3 of 200 cells" in text and "kB per cell" in text
    assert main(["verify", out, "--threads", "2"]) == 0
    assert "OK: 3 tiles, 530 cells" in capsys.readouterr().out
