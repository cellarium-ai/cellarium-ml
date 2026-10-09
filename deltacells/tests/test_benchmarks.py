# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests: the benchmark scripts run end to end on a tiny synthetic dataset."""

import importlib.util
import os
import sys

import pytest

torch = pytest.importorskip("torch")

BENCH = os.path.join(os.path.dirname(__file__), "..", "benchmarks")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(BENCH, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("bench") / "ds")
    assert (
        _load("make_synthetic").main(
            [root, "--n-cells", "2500", "--n-genes", "800", "--mean-nnz", "120", "--tile-size", "1000", "--level", "3"]
        )
        == 0
    )
    return root


def test_synthetic_dataset_is_sane(synthetic):
    from deltacells import open_dataset

    ds = open_dataset(synthetic)
    assert (ds.n_cells, ds.n_genes, ds.n_tiles) == (2500, 800, 3)
    assert ds.gene_order is not None
    b = ds.get_batch(range(100))
    assert 20 < b.nnz / 100 < 500


def test_bench_decode(synthetic, capsys):
    assert (
        _load("bench_decode").main(
            [synthetic, "--tiles", "1", "--threads", "1,2", "--repeats", "2", "--batch-size", "200"]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "decode,  1 thread(s)" in out and "gather 200 shuffled cells" in out


def test_bench_compression(synthetic, capsys):
    assert _load("bench_compression").main([synthetic, "--levels", "3", "--chunks", "1,2", "--threads", "2"]) == 0
    out = capsys.readouterr().out
    assert "sorted" in out and "random" in out


def test_bench_pipeline(synthetic, capsys):
    mod = _load("bench_pipeline")
    assert (
        mod.main(
            [
                synthetic,
                "--workers",
                "0,1",
                "--batch-size",
                "200",
                "--max-batches",
                "4",
                "--bandwidth-mbps",
                "500",
                "--latency-ms",
                "2",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "workers=0" in out and "workers=1" in out and "throttled" in out


def test_check_against_h5ad(tmp_path):
    anndata = pytest.importorskip("anndata")
    import numpy as np
    import pandas as pd
    from conftest import make_counts

    from deltacells.convert import convert_h5ad

    full = make_counts(300, 30, density=0.2, seed=70)
    os.makedirs(tmp_path / "in")
    for i, (lo, hi) in enumerate([(0, 120), (120, 300)]):
        anndata.AnnData(
            X=full[lo:hi],
            obs=pd.DataFrame(index=[str(j) for j in range(lo, hi)]),
            var=pd.DataFrame(index=[f"g{j}" for j in range(30)]),
        ).write_h5ad(tmp_path / "in" / f"s{i}.h5ad")
    out = str(tmp_path / "out")
    convert_h5ad(str(tmp_path / "in" / "*.h5ad"), out, tile_size=100, level=3, log=None)
    check = _load("check_against_h5ad")
    assert check.main([out, str(tmp_path / "in" / "*.h5ad")]) == 0
    # a modified dataset must be reported as a mismatch
    from deltacells import DatasetWriter

    bad = str(tmp_path / "bad")
    with DatasetWriter(bad, n_genes=30, tile_size=100, level=3) as w:
        for lo in range(0, 300, 100):
            m = full[lo : lo + 100].copy()
            if lo == 100:
                m.data[0] += 1
            w.add_tile(m)
    assert check.main([bad, str(tmp_path / "in" / "*.h5ad")]) == 1
    assert np.isfinite(full.data).all()


@pytest.fixture(scope="module")
def one_shard(tmp_path_factory):
    anndata = pytest.importorskip("anndata")
    import pandas as pd
    from conftest import make_counts

    path = str(tmp_path_factory.mktemp("shard") / "shard.h5ad")
    x = make_counts(800, 120, density=0.15, seed=90)
    anndata.AnnData(
        X=x, obs=pd.DataFrame(index=[str(i) for i in range(800)]), var=pd.DataFrame(index=[f"g{j}" for j in range(120)])
    ).write_h5ad(path)
    return path


def test_compare_formats_h5ad_and_deltacells(one_shard, capsys):
    pytest.importorskip("h5py")
    assert (
        _load("compare_formats").main(
            [one_shard, "--threads", "2", "--repeats", "1", "--level", "3", "--only", "h5ad,deltacells"]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert (
        "800 cells x 120 genes" in out and "h5ad (as is" in out and "deltacells (zstd 3)" in out and "| format |" in out
    )


def test_compare_formats_all_formats(one_shard, capsys):
    pytest.importorskip("tiledbsoma")
    pytest.importorskip("bpcells")
    assert _load("compare_formats").main([one_shard, "--threads", "2", "--repeats", "1", "--level", "3"]) == 0
    out = capsys.readouterr().out
    assert "TileDB-SOMA" in out and "BPCells" in out and "deltacells" in out


def test_parse_harness_output():
    mod = _load("compare_formats")
    r = mod.parse_harness_output(
        "warning: something\nRESULT nnz=10 compressed_bytes=99 read_ms=1.5 decode_ms_1=2.5 decode_ms_N=0.5 threads_N=8 correct=1\n"
    )
    assert r["nnz"] == 10 and r["read_ms"] == 1.5 and r["threads_N"] == 8 and r["correct"] == 1
    with pytest.raises(ValueError):
        mod.parse_harness_output("no result here")


_HARNESS = os.path.join(BENCH, "bpcells_decode", "build", "bpcells_decode_harness")


@pytest.mark.skipif(
    not os.path.exists(_HARNESS), reason="BPCells harness not built (benchmarks/bpcells_decode/build.sh)"
)
def test_bpcells_harness_decodes_correctly_and_is_reported(one_shard, capsys):
    pytest.importorskip("bpcells")
    assert _load("compare_formats").main([one_shard, "--threads", "2", "--repeats", "1", "--only", "bpcells"]) == 0
    out = capsys.readouterr().out
    assert "BPCells, Python bindings" in out and "BPCells, C++ decoder" in out


@pytest.fixture(scope="module")
def synthetic_with_obs(tmp_path_factory):
    pytest.importorskip("pyarrow")
    pytest.importorskip("pandas")
    root = str(tmp_path_factory.mktemp("benchobs") / "ds")
    args = [
        root,
        "--n-cells",
        "2500",
        "--n-genes",
        "300",
        "--mean-nnz",
        "60",
        "--tile-size",
        "1000",
        "--level",
        "3",
        "--obs-columns",
        "6",
    ]
    assert _load("make_synthetic").main(args) == 0
    return root


def test_synthetic_obs_dataset(synthetic_with_obs):
    from deltacells import open_dataset

    ds = open_dataset(synthetic_with_obs)
    assert (
        ds.obs is not None
        and ds.obs.columns[:4] == ["cell_type", "donor", "batch", "n_counts"]
        and "barcode" in ds.obs.columns
    )
    assert (
        ds.obs.kind("donor") == "category" and ds.obs.kind("qc_00") == "numeric" and ds.obs.kind("barcode") == "string"
    )


def test_bench_obs(synthetic_with_obs, capsys):
    assert (
        _load("bench_obs").main(
            [synthetic_with_obs, "--columns", "cell_type,qc_01", "--batch-size", "500", "--threads", "2"]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert (
        "localize ['cell_type', 'qc_01']" in out
        and "ranged reads" in out
        and "take(" in out
        and "per 100M cells" in out
    )
    assert (
        _load("bench_obs").main(
            [synthetic_with_obs, "--bandwidth-mbps", "200", "--latency-ms", "1", "--batch-size", "300"]
        )
        == 0
    )
