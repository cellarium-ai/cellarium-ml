# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("deltacells._core")

from cellarium.ml.api import CellariumData, create_deltacells_dataset  # noqa: E402
from cellarium.ml.api.cellariumdata import DeltaCellsLazyObs  # noqa: E402


def _obs(h5ad_paths: list[str]) -> pd.DataFrame:
    return pd.concat([ad.read_h5ad(p).obs for p in h5ad_paths], axis=0)


def _files(root: str) -> set[str]:
    return {os.path.relpath(os.path.join(r, f), root).replace(os.sep, "/") for r, _, fs in os.walk(root) for f in fs}


@pytest.fixture
def dcdata(deltacells_uri, deltacells_kwargs) -> CellariumData:
    return CellariumData.from_deltacells(
        deltacells_uri, batch_size=4, shuffle=False, deltacells_kwargs=deltacells_kwargs
    )


# --- create_deltacells_dataset ---------------------------------------------------------------------


def test_create_keeps_the_order_of_the_paths(h5ad_paths, tmp_path, deltacells_kwargs):
    reordered = [h5ad_paths[2], h5ad_paths[0], h5ad_paths[1]]
    uri = create_deltacells_dataset(reordered, str(tmp_path / "out"), tile_size=4, level=3, log=None)
    assert uri == str(tmp_path / "out")

    cdata = CellariumData.from_deltacells(uri, deltacells_kwargs=deltacells_kwargs)
    assert list(cdata.obs["cell_type"].index) == list(_obs(reordered).index)


def test_create_treats_paths_as_literals(make_h5ad_files, tmp_path):
    paths = make_h5ad_files(n_files=1)
    weird = str(tmp_path / "data[0].h5ad")
    os.rename(paths[0], weird)
    create_deltacells_dataset([weird], str(tmp_path / "out"), tile_size=4, level=3, log=None)


def test_create_rejects_bad_paths(h5ad_paths, tmp_path):
    out = str(tmp_path / "out")
    with pytest.raises(FileNotFoundError):
        create_deltacells_dataset([*h5ad_paths, str(tmp_path / "missing.h5ad")], out, log=None)
    with pytest.raises(ValueError, match="not supported"):
        create_deltacells_dataset(["s3://bucket/a.h5ad"], out, log=None)
    with pytest.raises(ValueError):
        create_deltacells_dataset([], out, log=None)
    assert not os.path.exists(out)


def test_create_refuses_to_overwrite_unless_asked(h5ad_paths, tmp_path):
    out = str(tmp_path / "out")
    create_deltacells_dataset(h5ad_paths, out, tile_size=4, level=3, log=None)
    with pytest.raises(FileExistsError):
        create_deltacells_dataset(h5ad_paths, out, tile_size=4, level=3, log=None)
    create_deltacells_dataset(h5ad_paths[:2], out, tile_size=4, level=3, log=None, overwrite=True)


class _RecordingFS:
    """An in-memory fsspec filesystem that records the order of the uploads and what is staged at each download."""

    def __init__(self):
        import fsspec

        self.fs = fsspec.filesystem("memory")
        self.puts: list[str] = []
        self.staged: list[int] = []  # the files in the scratch directory at each download, that one included

    def put_file(self, local, remote):
        self.puts.append(remote)
        self.fs.put_file(local, remote)

    def get_file(self, remote, local, **kwargs):
        self.fs.get_file(remote, local, **kwargs)
        self.staged.append(len([f for f in os.listdir(os.path.dirname(local)) if not f.endswith(".part")]))

    def __getattr__(self, name):
        return getattr(self.fs, name)


@pytest.fixture
def fake_gcs():
    """A fake bucket named ``bucket``, to be used as ``filesystem=`` with ``gs://bucket/...`` paths."""
    fs = _RecordingFS()
    yield fs
    if fs.fs.exists("bucket"):
        fs.fs.rm("bucket", recursive=True)


def _upload_inputs(fs: _RecordingFS, h5ad_paths: list[str]) -> list[str]:
    for i, path in enumerate(h5ad_paths):
        fs.fs.put_file(path, f"bucket/in/f{i}.h5ad")
    return [f"gs://bucket/in/f{i}.h5ad" for i in range(len(h5ad_paths))]


def _tiles(root: str) -> list[bytes]:
    names = sorted(f for f in _files(root) if f.startswith("tiles/"))
    return [open(os.path.join(root, f), "rb").read() for f in names]


def test_create_uploads_to_gcs_with_the_manifest_last(h5ad_paths, tmp_path, fake_gcs):
    local = str(tmp_path / "local")
    create_deltacells_dataset(h5ad_paths, local, tile_size=4, level=3, log=None)
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    uri = create_deltacells_dataset(
        h5ad_paths, "gs://bucket/prefix/", tile_size=4, level=3, log=None, filesystem=fake_gcs, staging_dir=str(scratch)
    )

    assert uri == "gs://bucket/prefix/"
    assert {p.removeprefix("bucket/prefix/") for p in fake_gcs.puts} == _files(local)
    assert fake_gcs.puts[-1] == "bucket/prefix/manifest.json"
    assert os.listdir(scratch) == []  # the scratch directory is cleaned up


def test_gcs_files_are_read_and_written_like_local_ones(h5ad_paths, tmp_path, fake_gcs):
    local = str(tmp_path / "local")
    create_deltacells_dataset(h5ad_paths, local, tile_size=4, level=3, log=None)
    urls = _upload_inputs(fake_gcs, h5ad_paths)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    kwargs: dict[str, Any] = dict(tile_size=4, level=3, log=None, filesystem=fake_gcs, staging_dir=str(scratch))

    remote_in = create_deltacells_dataset(urls, str(tmp_path / "remote_in"), **kwargs)
    create_deltacells_dataset(urls, "gs://bucket/out", **kwargs)

    assert _tiles(remote_in) == _tiles(local)
    assert {n.removeprefix("/bucket/out/") for n in fake_gcs.fs.find("bucket/out")} == _files(local)
    assert os.listdir(scratch) == []
    assert fake_gcs.fs.exists("bucket/in/f0.h5ad")  # the inputs are untouched


def test_gcs_files_are_staged_a_few_at_a_time(make_h5ad_files, tmp_path, fake_gcs):
    paths = make_h5ad_files(n_files=12, cells_per_file=5)
    urls = _upload_inputs(fake_gcs, paths)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    create_deltacells_dataset(
        urls, str(tmp_path / "out"), tile_size=4, level=3, log=None, filesystem=fake_gcs, staging_dir=str(scratch),
        workers=1, sort_genes_max_files=1,
    )  # fmt: skip
    assert len(fake_gcs.staged) == 12 + 1  # each file once, and one more for the gene order
    assert max(fake_gcs.staged) < 12  # never all of them at once
    assert os.listdir(scratch) == []


def test_upload_refuses_nonempty_prefix_unless_overwrite(h5ad_paths, tmp_path, fake_gcs):
    kwargs: dict[str, Any] = dict(
        tile_size=4, level=3, log=None, filesystem=fake_gcs, staging_dir=str(tmp_path), workers=1
    )
    create_deltacells_dataset(h5ad_paths, "gs://bucket/prefix", **kwargs)
    with pytest.raises(FileExistsError, match="overwrite"):
        create_deltacells_dataset(h5ad_paths[:1], "gs://bucket/prefix", **kwargs)

    fake_gcs.fs.pipe("bucket/prefix/tiles/tile_000099.dct", b"stale")  # a leftover of a bigger dataset
    create_deltacells_dataset(h5ad_paths[:1], "gs://bucket/prefix", overwrite=True, **kwargs)
    assert not fake_gcs.fs.exists("bucket/prefix/tiles/tile_000099.dct")
    assert fake_gcs.fs.exists("bucket/prefix/manifest.json")


def test_overwrite_never_deletes_what_is_not_a_dataset(h5ad_paths, tmp_path, fake_gcs):
    fake_gcs.fs.pipe("bucket/prefix/notes.txt", b"precious")
    fake_gcs.fs.pipe("bucket/prefix/manifest.json", b"{}")
    kwargs: dict[str, Any] = dict(tile_size=4, level=3, log=None, filesystem=fake_gcs, staging_dir=str(tmp_path))
    with pytest.raises(FileExistsError, match="notes.txt"):
        create_deltacells_dataset(h5ad_paths, "gs://bucket/prefix", overwrite=True, **kwargs)
    assert fake_gcs.fs.cat("bucket/prefix/notes.txt") == b"precious"
    assert fake_gcs.fs.cat("bucket/prefix/manifest.json") == b"{}"
    assert not fake_gcs.puts


def test_overwrite_never_deletes_the_inputs(h5ad_paths, tmp_path, fake_gcs):
    out = str(tmp_path / "out")
    with pytest.raises(ValueError, match="inside the output"):
        create_deltacells_dataset(h5ad_paths, os.path.dirname(h5ad_paths[0]), overwrite=True, log=None)
    urls = _upload_inputs(fake_gcs, h5ad_paths)
    with pytest.raises(ValueError, match="inside the output"):
        create_deltacells_dataset(urls, "gs://bucket/in", overwrite=True, log=None, filesystem=fake_gcs)
    assert all(os.path.exists(p) for p in h5ad_paths) and fake_gcs.fs.exists("bucket/in/f0.h5ad")
    assert not os.path.exists(out)


def _paths_with_big_counts(tmp_path, n_cells: int, big_in_second_file: bool, tag: str = "big") -> list[str]:
    """Two files of ``n_cells`` cells; in the second, every cell has a count of 70000 in MALAT1 (otherwise 2)."""
    import scipy.sparse as sp

    paths = []
    for i, big in enumerate([False, big_in_second_file]):
        x = np.tile(np.array([[1, 70_000 if (big and i == 1) else 2, 3]], dtype=np.float32), (n_cells, 1))
        obs = pd.DataFrame({"n": np.arange(n_cells)}, index=[f"f{i}_c{j}" for j in range(n_cells)])
        adata = ad.AnnData(X=sp.csr_matrix(x), obs=obs, var=pd.DataFrame(index=["a", "MALAT1", "c"]))
        paths.append(str(tmp_path / f"{tag}{i}.h5ad"))
        adata.write_h5ad(paths[-1])
    return paths


def test_counts_above_65535_are_stored_exactly(tmp_path):
    from deltacells import open_dataset

    paths = _paths_with_big_counts(tmp_path, n_cells=3, big_in_second_file=True)
    out = str(tmp_path / "out")
    create_deltacells_dataset(paths, out, tile_size=2, level=3, log=None)
    dataset = open_dataset(out)
    expected = np.vstack([ad.read_h5ad(p).X.toarray() for p in paths])
    assert np.array_equal(dataset[:].to_scipy().toarray()[:, np.argsort(dataset.gene_order)], expected)
    assert dataset[:].to_scipy().max() == 70_000


def test_too_many_counts_above_65535_are_refused(tmp_path):
    paths = _paths_with_big_counts(tmp_path, n_cells=1100, big_in_second_file=True)
    out = str(tmp_path / "out")
    with pytest.raises(ValueError, match="does not look like raw counts"):
        create_deltacells_dataset(paths, out, tile_size=1100, level=3, log=None, sort_genes=False)
    assert not os.path.exists(out)  # nothing is left behind


def test_a_failed_conversion_leaves_no_manifest_in_the_bucket(tmp_path, fake_gcs):
    bad = _paths_with_big_counts(tmp_path, n_cells=1100, big_in_second_file=True)  # the second tile fails
    urls = _upload_inputs(fake_gcs, bad)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with pytest.raises(ValueError, match="does not look like raw counts"):
        create_deltacells_dataset(
            urls, "gs://bucket/out", tile_size=1100, level=3, log=None, sort_genes=False, workers=1,
            filesystem=fake_gcs, staging_dir=str(scratch),
        )  # fmt: skip
    assert fake_gcs.fs.exists("bucket/out/tiles/tile_000000.dct")  # what was uploaded stays ...
    assert not fake_gcs.fs.exists("bucket/out/manifest.json")  # ... but it is not a dataset
    assert os.listdir(scratch) == []
    assert fake_gcs.fs.exists("bucket/in/f1.h5ad")

    good = _paths_with_big_counts(tmp_path, n_cells=1100, big_in_second_file=False, tag="good")
    urls = _upload_inputs(fake_gcs, good)  # the inputs are fixed ...
    create_deltacells_dataset(  # ... and, as the error says, running it again with overwrite replaces what is left
        urls, "gs://bucket/out", tile_size=1100, level=3, log=None, sort_genes=False, overwrite=True,
        filesystem=fake_gcs, staging_dir=str(scratch),
    )  # fmt: skip
    assert fake_gcs.fs.exists("bucket/out/manifest.json")


# --- CellariumData.from_deltacells ---------------------------------------------------------------


def test_constructor_needs_exactly_one_source(h5ad_paths, deltacells_uri):
    with pytest.raises(ValueError):
        CellariumData()
    with pytest.raises(ValueError):
        CellariumData(h5ad_paths=h5ad_paths, deltacells_uri=deltacells_uri)


def test_var_matches_the_h5ad_files(dcdata, h5ad_paths):
    expected = ad.read_h5ad(h5ad_paths[0]).var
    assert set(dcdata.var.index) == set(expected.index)
    assert list(dcdata.var.index) == list(dcdata.datamodule.var_names_g)  # stored order, consistent with the batches


def test_batches_match_the_h5ad_files_in_cell_order(dcdata, h5ad_paths):
    cdata_h5ad = CellariumData(h5ad_paths=h5ad_paths, batch_size=4, shuffle=False)
    expected_x = {}
    for batch in cdata_h5ad.datamodule.train_dataloader():
        for name, row in zip(batch["obs_names_n"], batch["x_ng"].to_dense().numpy()):
            expected_x[name] = pd.Series(row, index=batch["var_names_g"])

    names = []
    for batch in dcdata.datamodule.train_dataloader():
        for name, row in zip(batch["obs_names_n"], batch["x_ng"].to_dense().numpy()):
            names.append(name)
            pd.testing.assert_series_equal(
                pd.Series(row, index=batch["var_names_g"]).reindex(expected_x[name].index),
                expected_x[name],
                check_names=False,
            )
    assert names == list(_obs(h5ad_paths).index)


def test_obs_queries(dcdata, h5ad_paths):
    expected = _obs(h5ad_paths)
    obs = dcdata.obs
    assert isinstance(obs, DeltaCellsLazyObs)
    assert len(obs) == len(expected)
    assert obs.columns == ["cell_type", "n_counts"]

    cell_type = obs["cell_type"]
    assert list(cell_type.index) == list(expected.index)
    assert list(cell_type.astype(str)) == list(expected["cell_type"].astype(str))
    assert obs[["cell_type", "n_counts"]].shape == (len(expected), 2)

    names = [expected.index[7], expected.index[2]]
    selected = obs.loc[names]
    assert list(selected.index) == names
    assert list(selected["n_counts"]) == list(expected.loc[names, "n_counts"])
    assert list(obs.loc[names, "n_counts"].columns) == ["n_counts"]
    with pytest.raises(KeyError):
        obs.loc[["no-such-cell"]]

    batches = list(obs.iter_batches(batch_size=4, columns=["n_counts"]))
    assert [len(b) for b in batches] == [4, 4, 4, 3]
    assert list(pd.concat(batches)["n_counts"]) == list(expected["n_counts"])

    frame = obs.to_frame()
    assert list(frame.index) == list(expected.index)
    assert list(frame.columns) == ["cell_type", "n_counts"]


def test_dataset_without_obs_names_is_rejected(h5ad_paths, tmp_path, deltacells_kwargs):
    from deltacells.convert import convert_h5ad

    convert_h5ad(h5ad_paths, str(tmp_path / "out"), tile_size=4, level=3, log=None, obs_names=False)
    with pytest.raises(KeyError, match="obs_names"):
        CellariumData.from_deltacells(str(tmp_path / "out"), deltacells_kwargs=deltacells_kwargs)


def test_datamodule_helpers(dcdata):
    datamodule = dcdata.datamodule
    assert list(datamodule.var_names_g) == list(datamodule.dadc.var_names)
    assert datamodule.obs_key_nunique("cell_type") == 3
    with pytest.raises(ValueError):
        datamodule.obs_key_nunique("no_such_column")


def test_obs_columns_become_batch_entries(deltacells_uri, deltacells_kwargs):
    from cellarium.ml.utilities.data import categories_to_codes

    obs_columns = {"y_n": ("cell_type", categories_to_codes)}
    cdata = CellariumData.from_deltacells(
        deltacells_uri,
        obs_columns=obs_columns,
        total_mrna_umis_column="n_counts",
        batch_size=4,
        deltacells_kwargs=deltacells_kwargs,
    )
    assert obs_columns == {"y_n": ("cell_type", categories_to_codes)}  # the argument is not modified
    batch = next(iter(cdata.datamodule.train_dataloader()))
    assert batch["y_n"].shape == (4,)
    assert batch["total_mrna_umis_n"].shape == (4,)


def test_hvg_and_obs_computed(dcdata):
    dcdata.hvg = dcdata.var.index[:2].to_numpy()
    assert dcdata.hvg is not None and int(dcdata.hvg.sum()) == 2

    dcdata.obs_computed["flag"] = pd.Series(np.zeros(15, dtype=bool))
    with pytest.raises(ValueError):
        dcdata.obs_computed["bad"] = np.zeros(14)
    assert "obs_computed keys: ['flag']" in repr(dcdata)
