# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellarium.ml.api import _view_collections
from cellarium.ml.api._view_collections import RamCollection, load_into_ram, unwrap_collection
from cellarium.ml.api.cellariumdata import CellariumData, CellariumDataView
from cellarium.ml.api.deltacells_store import write_collection_to_deltacells
from cellarium.ml.data import DistributedDeltaCellsCollection
from cellarium.ml.utilities.data import AnnDataField, categories_to_codes

# the `cdata` fixture (tests/api/conftest.py): 40 cells in two h5ad files of 20, as h5ad files or as a deltacells
# dataset in tiles of 15, with obs columns `cell_type` and `n_counts`

POSITIONS = np.array([0, 1, 5, 6, 7, 16, 19, 20, 21, 22, 30, 33, 39])


def _epoch(cdata: CellariumData) -> dict:
    names, rows = [], []
    for batch in cdata.datamodule.train_dataloader():
        names.extend(batch["obs_names_n"])
        rows.append(batch["x_ng"].to_dense().numpy())
    return {"names": names, "x": np.concatenate(rows)}


def _label_view(cdata, positions, labels, key="subtype"):
    view = cdata[positions]
    view.obs_computed[key] = pd.Series(labels, index=view.obs.index)
    return view


# ---- to_ram ----


def test_to_ram_returns_a_view_of_the_same_cells_that_reads_from_memory(cdata):
    view = cdata[POSITIONS]
    ram = view.to_ram()
    assert isinstance(ram, CellariumDataView)
    assert ram.root is cdata and ram.parent is view
    assert ram.root_indices.tolist() == POSITIONS.tolist()
    assert ram.n_obs == len(POSITIONS)
    collection = ram.datamodule.dadc
    assert isinstance(collection, RamCollection)
    assert collection.limits == [len(POSITIONS)] and collection.max_num_workers == 0
    assert collection.nbytes > 0 and "RamCollection" in repr(collection)
    assert ram.obs.index.tolist() == view.obs.index.tolist()


def test_a_view_in_memory_yields_the_same_cells_as_the_view_it_came_from(cdata):
    view = cdata[POSITIONS]
    ram = view.to_ram()
    source, copy = view.datamodule.dadc, ram.datamodule.dadc
    order = [4, 0, 12, 7, 7, 3]
    expected, got = source[order], copy[order]
    assert np.array_equal(np.asarray(got.X.todense()), np.asarray(expected.X.todense()))
    assert got.X.dtype == np.float32
    assert list(got.obs_names) == list(expected.obs_names)
    assert list(got.var_names) == list(expected.var_names)
    assert got.var.equals(expected.var)
    assert np.array_equal(np.asarray(copy[2:6].X.todense()), np.asarray(source[2:6].X.todense()))
    assert np.array_equal(copy[[12]].X.toarray(), source[[12]].X.toarray())
    # the obs columns have the same dtypes and categories, so that the codes of the categories are the same
    for column in ("cell_type", "n_counts"):
        assert got.obs[column].dtype == expected.obs[column].dtype
    field = AnnDataField(attr="obs", key="cell_type", convert_fn=categories_to_codes)
    assert np.array_equal(field(got), field(expected))
    assert list(copy.obs_categories("cell_type")) == list(source.obs_categories("cell_type"))
    assert copy.obs_key_nunique("cell_type") == source.obs_key_nunique("cell_type")


def test_an_epoch_over_a_view_in_memory_yields_each_cell_once(cdata):
    view = cdata[POSITIONS]
    ram = view.to_ram()
    ram.datamodule.batch_size = 4
    ram.datamodule.setup("fit")
    epoch = _epoch(ram)
    assert sorted(epoch["names"]) == sorted(cdata.obs.index[POSITIONS])
    whole = view.datamodule.dadc[np.arange(len(POSITIONS))]
    reference = dict(zip(whole.obs_names, whole.X.toarray()))
    assert all(np.array_equal(row, reference[name]) for name, row in zip(epoch["names"], epoch["x"]))


def test_to_ram_of_the_whole_data_is_a_view_of_all_cells(cdata):
    ram = cdata.to_ram()
    assert ram.n_obs == 40 and ram.root_indices.tolist() == list(range(40))
    assert ram.parent is cdata


def test_data_in_memory_is_not_loaded_again_and_views_of_it_share_the_memory(cdata):
    ram = cdata[POSITIONS].to_ram()
    assert ram.to_ram() is ram
    sub = ram[[1, 3, 4]]
    assert isinstance(unwrap_collection(sub.datamodule.dadc), RamCollection)
    assert unwrap_collection(sub.datamodule.dadc) is unwrap_collection(ram.datamodule.dadc)
    assert sub.to_ram() is sub
    assert sub.root_indices.tolist() == POSITIONS[[1, 3, 4]].tolist()
    assert sub.datamodule.num_workers == 0


def test_a_view_in_memory_starts_with_snapshots_and_no_modules(cdata):
    cdata.obsm["emb"] = np.arange(80, dtype=np.float32).reshape(40, 2)
    cdata.obs_computed["score"] = np.arange(40) * 1.0
    cdata.hvg = np.arange(len(cdata.var)) % 2 == 0
    ram = cdata[POSITIONS].to_ram()
    assert np.array_equal(ram.obsm["emb"], cdata.obsm["emb"][POSITIONS])
    assert np.array_equal(ram.obs_computed["score"], POSITIONS * 1.0)
    assert ram.hvg is None and len(ram.trained_modules) == 0


def test_update_from_works_from_a_view_in_memory(cdata):
    view = cdata[POSITIONS]
    ram = view.to_ram()
    ram.obs_computed["s"] = pd.Series(np.arange(len(POSITIONS)).astype(str), index=ram.obs.index)
    view.obs_computed.update_from(ram, "s")
    cdata.obs_computed.update_from(ram, "s")
    assert cdata.obs_computed["s"].iloc[POSITIONS].tolist() == [str(i) for i in range(len(POSITIONS))]
    assert cdata.obs_computed["s"].notna().sum() == len(POSITIONS)
    assert view.obs_computed["s"].tolist() == [str(i) for i in range(len(POSITIONS))]


def test_a_view_in_memory_cannot_use_dataloader_workers(cdata):
    cdata.datamodule.num_workers = 2
    ram = cdata[POSITIONS].to_ram()
    assert ram.datamodule.num_workers == 0  # whatever the parent has
    ram.datamodule.num_workers = 2
    with pytest.raises(ValueError, match="at most 0"):
        ram.datamodule.train_dataloader()
    ram.datamodule.num_workers = 0
    ram.datamodule.train_dataloader()
    # data that is not in memory has no such limit
    assert cdata[POSITIONS].datamodule.num_workers == 2


# ---- the memory guard ----


def test_to_ram_refuses_cells_that_do_not_fit_and_loads_nothing(cdata, monkeypatch):
    view = cdata[POSITIONS]
    loaded = []
    real = _view_collections.RamCollection.__init__

    def recording_init(self, *args, **kwargs):
        loaded.append(1)
        real(self, *args, **kwargs)

    monkeypatch.setattr(_view_collections.RamCollection, "__init__", recording_init)
    with pytest.raises(MemoryError, match="would need about .* but at most .* may be used"):
        view.to_ram(max_gb=1e-9)
    assert not loaded


def test_the_default_limit_is_half_of_the_available_memory(cdata, monkeypatch):
    view = cdata[POSITIONS]
    monkeypatch.setattr("cellarium.ml.api.cellariumdata.available_memory_bytes", lambda: 2000)
    with pytest.raises(MemoryError, match="at most 0.00 GiB"):
        view.to_ram()
    monkeypatch.setattr("cellarium.ml.api.cellariumdata.available_memory_bytes", lambda: 10 * 2**30)
    assert isinstance(view.to_ram().datamodule.dadc, RamCollection)


def test_available_memory_is_a_positive_number():
    assert _view_collections.available_memory_bytes() > 0


@pytest.fixture
def uneven_collection():
    """Cells whose first 1000 are sparse and the rest dense, so a sample from the start underestimates the memory."""
    rng = np.random.default_rng(0)
    n, g = 4000, 50
    dense = (rng.random((n, g)) < 0.9) * rng.integers(1, 5, size=(n, g))
    dense[:1000] = 0
    dense[:1000, 0] = 1
    x = sp.csr_matrix(dense.astype(np.float32))
    obs = pd.DataFrame({"label": pd.Categorical(rng.choice(["a", "b"], n))}, index=[f"c{i}" for i in range(n)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(g)])
    return RamCollection(x, obs, var, var.index.to_numpy()), dense


def test_loading_grows_its_arrays_when_the_estimate_is_too_low(uneven_collection):
    source, dense = uneven_collection
    copy = load_into_ram(source, max_bytes=2**30, chunk_size=500, n_samples=1, sample_size=100, progress=False)
    assert np.array_equal(copy.x.toarray(), dense)
    assert copy.x.indices.dtype == np.int32 and copy.x.data.dtype == np.float32
    assert copy.obs.index.tolist() == source.obs.index.tolist()
    assert copy.obs["label"].dtype == source.obs["label"].dtype
    assert copy.x.data.shape[0] == copy.x.nnz  # trimmed to what was read


def test_loading_stops_when_it_runs_past_the_limit_though_the_estimate_fit(uneven_collection):
    source, dense = uneven_collection
    # the sparse start suggests about 0.3 MB (mostly the obs), the real counts take over 1 MB
    limit = 600_000
    assert limit < source.x.nnz * 8
    with pytest.raises(MemoryError, match="needs more than"):
        load_into_ram(source, max_bytes=limit, chunk_size=500, n_samples=1, sample_size=100, progress=False)


# ---- to_deltacells ----


@pytest.fixture
def written(cdata, tmp_path, deltacells_kwargs):
    """A view of some cells of ``cdata`` and the view that ``to_deltacells`` returns for it."""
    pytest.importorskip("deltacells._core")
    view = cdata[POSITIONS]
    new = view.to_deltacells(str(tmp_path / "subset"), tile_size=4, level=3, deltacells_kwargs=deltacells_kwargs)
    return view, new


def _dense_by_gene(dadc, cells, genes):
    """The counts of ``cells`` of the collection ``dadc``, with the columns in the order ``genes``."""
    batch = dadc[cells]
    frame = pd.DataFrame(batch.X.toarray(), columns=list(batch.var_names))
    return frame[list(genes)].to_numpy()


def test_to_deltacells_returns_a_view_of_the_same_cells_that_reads_the_new_dataset(cdata, written, tmp_path):
    view, new = written
    assert isinstance(new, CellariumDataView)
    assert new.root is cdata and new.parent is view
    assert new.root_indices.tolist() == POSITIONS.tolist()
    assert new.n_obs == len(POSITIONS)
    assert new.deltacells_uri == str(tmp_path / "subset")
    assert isinstance(unwrap_collection(new.datamodule.dadc), DistributedDeltaCellsCollection)
    assert len(new.datamodule.dadc) == len(POSITIONS)
    assert new.obs.index.tolist() == view.obs.index.tolist()


def test_the_new_dataset_holds_the_same_cells(written):
    view, new = written
    cells = np.arange(len(POSITIONS))
    old, copy = view.datamodule.dadc[cells], new.datamodule.dadc[cells]
    assert list(copy.obs_names) == list(old.obs_names)
    genes = list(old.var_names)
    assert sorted(copy.var_names) == sorted(genes)
    new_counts = _dense_by_gene(new.datamodule.dadc, cells, genes)
    assert np.array_equal(new_counts, _dense_by_gene(view.datamodule.dadc, cells, genes))
    for column in ("cell_type", "n_counts"):
        assert copy.obs[column].astype(str).tolist() == old.obs[column].astype(str).tolist()


def test_the_new_dataset_keeps_the_categories_of_the_data_including_unused_ones(cdata, tmp_path, deltacells_kwargs):
    pytest.importorskip("deltacells._core")
    types = cdata.obs["cell_type"].astype(str).to_numpy()
    only_t = cdata[types == "T cell"]
    assert 0 < only_t.n_obs < 40
    new = only_t.to_deltacells(str(tmp_path / "t_cells"), tile_size=4, level=3, deltacells_kwargs=deltacells_kwargs)
    old_categories = list(only_t.datamodule.dadc.obs_categories("cell_type"))
    assert list(new.datamodule.dadc.obs_categories("cell_type")) == old_categories
    assert new.datamodule.dadc.obs_key_nunique("cell_type") == len(old_categories)
    field = AnnDataField(attr="obs", key="cell_type", convert_fn=categories_to_codes)
    cells = np.arange(only_t.n_obs)
    assert np.array_equal(field(new.datamodule.dadc[cells]), field(only_t.datamodule.dadc[cells]))


def test_an_epoch_over_the_new_dataset_yields_each_cell_once(written):
    view, new = written
    new.datamodule.batch_size = 4
    new.datamodule.setup("fit")
    epoch = _epoch(new)
    assert sorted(epoch["names"]) == sorted(view.obs.index)


def test_update_from_works_from_the_view_of_the_new_dataset(cdata, written):
    view, new = written
    new.obs_computed["s"] = pd.Series(np.arange(len(POSITIONS)).astype(str), index=new.obs.index)
    cdata.obs_computed.update_from(new, "s")
    assert cdata.obs_computed["s"].iloc[POSITIONS].tolist() == [str(i) for i in range(len(POSITIONS))]


def test_the_new_dataset_can_be_opened_on_its_own_as_a_new_root(cdata, written, tmp_path, deltacells_kwargs):
    view, new = written
    fresh = CellariumData.from_deltacells(new.deltacells_uri, deltacells_kwargs=deltacells_kwargs)
    assert fresh.root is fresh and fresh.root is not cdata
    assert fresh.n_obs == len(POSITIONS)
    assert fresh.obs.index.tolist() == view.obs.index.tolist()
    fresh.obs_computed["x"] = np.arange(len(POSITIONS)) * 1.0
    with pytest.raises(ValueError, match="same root"):
        cdata.obs_computed.update_from(fresh, "x")


def test_to_deltacells_works_from_data_in_memory_and_from_the_whole_data(cdata, tmp_path, deltacells_kwargs):
    pytest.importorskip("deltacells._core")
    ram = cdata[POSITIONS].to_ram()
    new = ram.to_deltacells(str(tmp_path / "from_ram"), tile_size=5, level=3, deltacells_kwargs=deltacells_kwargs)
    assert new.root_indices.tolist() == POSITIONS.tolist() and new.parent is ram
    assert new.obs.index.tolist() == ram.obs.index.tolist()
    everything = cdata.to_deltacells(str(tmp_path / "all"), tile_size=15, level=3, deltacells_kwargs=deltacells_kwargs)
    assert everything.n_obs == 40 and everything.root_indices.tolist() == list(range(40))


def test_sort_genes_false_keeps_the_gene_order(cdata, tmp_path, deltacells_kwargs):
    pytest.importorskip("deltacells._core")
    view = cdata[POSITIONS]
    new = view.to_deltacells(
        str(tmp_path / "unsorted"), tile_size=4, level=3, sort_genes=False, deltacells_kwargs=deltacells_kwargs
    )
    assert list(new.datamodule.dadc.var_names) == list(view.datamodule.dadc.var_names)
    cells = np.arange(len(POSITIONS))
    assert np.array_equal(new.datamodule.dadc[cells].X.toarray(), view.datamodule.dadc[cells].X.toarray())


def test_to_deltacells_refuses_to_overwrite_unless_asked(cdata, written, tmp_path, deltacells_kwargs):
    view, new = written
    with pytest.raises(FileExistsError):
        view.to_deltacells(new.deltacells_uri, tile_size=4, level=3)
    assert os.path.exists(os.path.join(new.deltacells_uri, "manifest.json"))  # what was there is untouched
    again = cdata[:6].to_deltacells(
        new.deltacells_uri, tile_size=4, level=3, overwrite=True, deltacells_kwargs=deltacells_kwargs
    )
    assert again.n_obs == 6


def test_data_that_are_not_counts_are_refused_and_nothing_is_left_behind(tmp_path):
    pytest.importorskip("deltacells._core")
    rng = np.random.default_rng(0)
    paths = []
    counts, not_counts = rng.poisson(3.0, size=(6, 4)), rng.random((6, 4))
    for i, values in enumerate([counts.astype(np.float32), not_counts.astype(np.float32)]):
        obs = pd.DataFrame({"n": np.arange(6)}, index=[f"f{i}_c{j}" for j in range(6)])
        adata = ad.AnnData(X=sp.csr_matrix(values), obs=obs, var=pd.DataFrame(index=list("abcd")))
        paths.append(str(tmp_path / f"f{i}.h5ad"))
        adata.write_h5ad(paths[-1])
    cdata = CellariumData(h5ad_paths=paths)
    out = str(tmp_path / "out")
    with pytest.raises(ValueError, match="integer counts"):
        cdata.to_deltacells(out, tile_size=4, level=3)
    assert not os.path.exists(out)


def test_a_collection_is_written_to_gcs_with_the_manifest_last(cdata, tmp_path):
    pytest.importorskip("deltacells._core")
    from tests.api.test_deltacells import _RecordingFS

    fs = _RecordingFS()
    uri = write_collection_to_deltacells(
        cdata[POSITIONS].datamodule.dadc,
        "gs://bucket/prefix",
        tile_size=4,
        level=3,
        filesystem=fs,
        staging_dir=str(tmp_path),
    )
    assert uri == "gs://bucket/prefix"
    assert fs.puts[-1] == "bucket/prefix/manifest.json"
    assert not [d for d in os.listdir(tmp_path) if d.startswith("cellarium_deltacells_")]  # staging cleaned up
    fs.fs.rm("bucket", recursive=True)
