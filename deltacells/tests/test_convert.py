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
