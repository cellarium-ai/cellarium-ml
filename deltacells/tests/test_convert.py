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


def test_sort_genes_orders_by_first_shard_totals(tmp_path, full):
    write_shards(tmp_path / "in", [200, 200, 130], full)
    out = str(tmp_path / "out")
    convert_h5ad(str(tmp_path / "in" / "*.h5ad"), out, tile_size=150, n_chunks=2, level=3, threads=1, log=None)
    ds = open_dataset(out)
    first = np.asarray(full[:200].sum(0)).ravel()
    order = ds.gene_order
    assert sorted(order) == list(range(G)) and (np.diff(first[order]) <= 0).all()
    assert same(ds[:].to_scipy(), full[:, order])
    assert ds.var_names == [f"gene{j}" for j in order]
    assert ds.manifest.metadata["sort_genes"] is True and ds.manifest.metadata["n_source_files"] == 3


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
