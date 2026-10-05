# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os

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
    return {
        os.path.relpath(os.path.join(r, f), root).replace(os.sep, "/") for r, _, fs in os.walk(root) for f in fs
    }


@pytest.fixture
def dcdata(deltacells_uri, deltacells_kwargs) -> CellariumData:
    return CellariumData.from_deltacells(deltacells_uri, batch_size=4, shuffle=False, deltacells_kwargs=deltacells_kwargs)


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
    with pytest.raises(ValueError, match="local"):
        create_deltacells_dataset(["gs://bucket/a.h5ad"], out, log=None)
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
    """An in-memory fsspec filesystem that records the order of the uploads."""

    def __init__(self):
        import fsspec

        self.fs = fsspec.filesystem("memory")
        self.puts: list[str] = []

    def put_file(self, local, remote):
        self.puts.append(remote)
        self.fs.put_file(local, remote)

    def __getattr__(self, name):
        return getattr(self.fs, name)


def test_create_uploads_to_gcs_with_the_manifest_last(h5ad_paths, tmp_path):
    local = str(tmp_path / "local")
    create_deltacells_dataset(h5ad_paths, local, tile_size=4, level=3, log=None)
    fs = _RecordingFS()

    uri = create_deltacells_dataset(
        h5ad_paths, "gs://bucket/prefix/", tile_size=4, level=3, log=None, filesystem=fs, staging_dir=str(tmp_path)
    )

    assert uri == "gs://bucket/prefix/"
    assert {p.removeprefix("bucket/prefix/") for p in fs.puts} == _files(local)
    assert fs.puts[-1] == "bucket/prefix/manifest.json"
    assert not [d for d in os.listdir(tmp_path) if d.startswith("cellarium_deltacells_")]  # staging cleaned up
    fs.fs.rm("bucket", recursive=True)


def test_upload_refuses_nonempty_prefix_unless_overwrite(h5ad_paths, tmp_path):
    fs = _RecordingFS()
    fs.fs.pipe("bucket/prefix/stale.txt", b"old")
    kwargs = dict(tile_size=4, level=3, log=None, filesystem=fs, staging_dir=str(tmp_path))
    with pytest.raises(FileExistsError):
        create_deltacells_dataset(h5ad_paths, "gs://bucket/prefix", **kwargs)
    assert not fs.puts

    create_deltacells_dataset(h5ad_paths, "gs://bucket/prefix", overwrite=True, **kwargs)
    assert not fs.fs.exists("bucket/prefix/stale.txt")
    assert fs.fs.exists("bucket/prefix/manifest.json")
    fs.fs.rm("bucket", recursive=True)


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
