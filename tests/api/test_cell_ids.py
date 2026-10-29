# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellarium.ml.api.cellariumdata import CellariumData

# the `cdata` fixture (tests/api/conftest.py): 40 cells in two h5ad files of 20, as h5ad files or as a deltacells
# dataset in tiles of 15, with obs columns `cell_type` and `n_counts`

POSITIONS = np.array([0, 1, 5, 6, 7, 16, 19, 20, 21, 22, 30, 33, 39])


def _compact(view, tmp_path, deltacells_kwargs, name="compact") -> CellariumData:
    """The cells of ``view`` as a dataset of their own, opened in a 'new session' (a new root)."""
    pytest.importorskip("deltacells._core")
    uri = str(tmp_path / name)
    view.to_deltacells(uri, tile_size=4, level=3, deltacells_kwargs=deltacells_kwargs)
    return CellariumData.from_deltacells(uri, deltacells_kwargs=deltacells_kwargs, shuffle=False)


# ---- the IDs of the cells ----


def test_the_cell_ids_of_a_source_are_the_positions_of_its_cells(cdata):
    assert cdata.cell_ids.tolist() == list(range(40))
    assert cdata.cell_ids.name == "cellarium_cell_id"
    assert "cellarium_cell_id" not in cdata.obs.columns  # nothing is stored for them in a source
    # obs reads are indexed by them, without reading the names
    assert cdata.obs["cell_type"].index.equals(cdata.cell_ids)
    assert cdata.obs[["cell_type", "n_counts"]].index.equals(cdata.cell_ids)
    assert len(cdata.obs_names) == 40 and cdata.obs_names[0] == "file0_cell0"


def test_a_view_has_the_ids_of_its_cells_in_the_root(cdata):
    view = cdata[POSITIONS]
    assert view.cell_ids.tolist() == POSITIONS.tolist()
    assert view.obs["n_counts"].index.tolist() == POSITIONS.tolist()
    assert view.obs_names.tolist() == cdata.obs_names[POSITIONS].tolist()
    assert view[[2, 4]].cell_ids.tolist() == POSITIONS[[2, 4]].tolist()


def test_to_deltacells_stores_the_ids_and_a_dataset_opened_on_its_own_has_them(cdata, tmp_path, deltacells_kwargs):
    view = cdata[POSITIONS]
    compact = _compact(view, tmp_path, deltacells_kwargs)

    assert compact.n_obs == len(POSITIONS)
    assert compact.cell_ids.tolist() == POSITIONS.tolist()  # not 0, 1, 2, ...
    assert "cellarium_cell_id" in compact.obs.columns  # it is an obs column there
    assert compact.obs["n_counts"].index.tolist() == POSITIONS.tolist()
    assert compact.obs_names.tolist() == cdata.obs_names[POSITIONS].tolist()
    assert compact.obs["n_counts"].tolist() == cdata.obs["n_counts"].to_numpy()[POSITIONS].tolist()

    # a view of it has the IDs of its cells, which are not their positions
    inner = compact[[1, 2, 5]]
    assert inner.cell_ids.tolist() == POSITIONS[[1, 2, 5]].tolist()
    assert inner.root_indices.tolist() == [1, 2, 5]
    assert compact.select_cell_ids(POSITIONS[[5, 1]]).root_indices.tolist() == [1, 5]
    with pytest.raises(KeyError, match="not among"):
        compact.select_cell_ids([2])  # a cell of the original data that is not in this one


def test_compacting_a_compacted_dataset_keeps_the_original_ids(cdata, tmp_path, deltacells_kwargs):
    compact = _compact(cdata[POSITIONS], tmp_path, deltacells_kwargs)
    inner = _compact(compact[[0, 3, 4, 8, 9]], tmp_path, deltacells_kwargs, name="inner")
    assert inner.cell_ids.tolist() == POSITIONS[[0, 3, 4, 8, 9]].tolist()


def test_the_cells_of_a_compacted_dataset_are_selected_by_cell_id_masks(cdata, tmp_path, deltacells_kwargs):
    compact = _compact(cdata[POSITIONS], tmp_path, deltacells_kwargs)
    # a mask computed on the original data selects the cells that the compacted dataset has of those
    mask = pd.Series(np.arange(40) % 2 == 0, index=cdata.cell_ids)
    assert compact[mask].cell_ids.tolist() == [i for i in POSITIONS if i % 2 == 0]
    # and a mask from the obs of the compacted dataset is indexed by its IDs, so it works on either
    t_cells = compact.obs["cell_type"] == "T cell"
    assert compact[t_cells].cell_ids.tolist() == compact.cell_ids[t_cells.to_numpy()].tolist()
    assert cdata[t_cells].cell_ids.tolist() == compact.cell_ids[t_cells.to_numpy()].tolist()


# ---- labels from an AnnData ----


def _clustered(cdata, positions=None, key="leiden"):
    """The AnnData of the cells (all of them by default) with some made-up clusters in ``obs[key]``."""
    adata = (cdata if positions is None else cdata[positions]).to_anndata()
    adata.obs[key] = pd.Categorical(["c" + str(i % 3) for i in range(adata.n_obs)])
    return adata


def test_labels_of_an_anndata_are_assigned_to_their_cells(cdata):
    adata = _clustered(cdata, POSITIONS)
    cdata.update_labels_from_anndata(adata, "leiden")

    labels = cdata.obs_computed["leiden"]
    assert isinstance(labels, pd.Series) and labels.index.equals(cdata.cell_ids)
    assert isinstance(labels.dtype, pd.CategoricalDtype)
    assert labels.iloc[POSITIONS].tolist() == adata.obs["leiden"].tolist()
    assert labels.drop(POSITIONS).isna().all() and len(labels) == 40


def test_labels_go_under_the_key_that_is_given_and_are_made_if_missing(cdata):
    adata = _clustered(cdata)
    cdata.update_labels_from_anndata(adata, "leiden", as_key="clusters_1.0")
    assert "leiden" not in cdata.obs_computed
    assert cdata.obs_computed["clusters_1.0"].tolist() == adata.obs["leiden"].tolist()


def test_cells_that_scanpy_removed_are_missing(cdata):
    adata = _clustered(cdata)
    kept = adata[adata.obs["n_counts"] > 3000].copy()  # as if filtered by the user
    assert 0 < kept.n_obs < 40
    cdata.update_labels_from_anndata(kept, "leiden")
    labels = cdata.obs_computed["leiden"]
    assert labels.notna().sum() == kept.n_obs
    assert labels[kept.obs["cellarium_cell_id"].to_numpy()].tolist() == kept.obs["leiden"].tolist()


def test_the_labels_of_several_anndatas_fill_one_entry(cdata):
    first = _clustered(cdata, np.array([0, 1, 2, 3]))
    second = _clustered(cdata, np.array([2, 3, 10]))
    cdata.update_labels_from_anndata(first, "leiden", as_key="all")
    cdata.update_labels_from_anndata(second, "leiden", as_key="all")  # replaces the values of cells 2 and 3
    labels = cdata.obs_computed["all"]
    assert labels.notna().sum() == 5
    assert labels.iloc[[0, 1]].tolist() == first.obs["leiden"].tolist()[:2]
    assert labels.iloc[[2, 3, 10]].tolist() == second.obs["leiden"].tolist()
    with pytest.raises(ValueError, match="already"):
        cdata.update_labels_from_anndata(second, "leiden", as_key="all", overwrite=False)


def test_the_labels_of_an_anndata_of_a_view_are_assigned_to_the_view_and_the_root(cdata):
    view = cdata[POSITIONS]
    adata = _clustered(view)
    view.update_labels_from_anndata(adata, "leiden", as_key="sub")
    cdata.update_labels_from_anndata(adata, "leiden", as_key="sub")  # the cell IDs are those of the root
    assert view.obs_computed["sub"].tolist() == adata.obs["leiden"].tolist()
    assert cdata.obs_computed["sub"].iloc[POSITIONS].tolist() == adata.obs["leiden"].tolist()
    # ... and the view can then be selected with the labels of the root
    assert cdata[cdata.obs_computed["sub"] == "c0"].cell_ids.tolist() == POSITIONS[::3].tolist()


def test_update_labels_from_anndata_checks_what_it_is_given(cdata):
    adata = _clustered(cdata)
    with pytest.raises(KeyError, match="no 'nothing'"):
        cdata.update_labels_from_anndata(adata, "nothing")
    bare = ad.AnnData(X=np.zeros((3, 2)), obs=pd.DataFrame({"x": ["a", "b", "c"]}, index=["a", "b", "c"]))
    with pytest.raises(KeyError, match="cellarium_cell_id"):
        cdata.update_labels_from_anndata(bare, "x")
    adata.obs["cellarium_cell_id"] = 0
    with pytest.raises(ValueError, match="not unique"):
        cdata.update_labels_from_anndata(adata, "leiden")
    adata.obs["cellarium_cell_id"] = np.arange(40) + 1000
    with pytest.raises(KeyError, match="not among"):
        cdata.update_labels_from_anndata(adata, "leiden")
    assert "leiden" not in cdata.obs_computed


# ---- saving labels, and bringing them back to the original data ----


def test_labels_are_saved_and_loaded_by_cell_id(cdata, tmp_path):
    cdata.obs_computed["a"] = pd.Series(pd.Categorical(["x", "y"] * 20), index=cdata.cell_ids).where(np.arange(40) < 10)
    cdata.obs_computed["score"] = np.arange(40) * 0.5
    path = str(tmp_path / "labels.parquet")
    cdata.save_labels(["a", "score"], path)

    saved = pd.read_parquet(path)
    assert list(saved.columns) == ["cellarium_cell_id", "a", "score"]
    assert len(saved) == 40  # `score` has a value for every cell

    cdata.save_labels("a", path)
    assert len(pd.read_parquet(path)) == 10  # only the cells that have a label are saved

    cdata.obs_computed.clear()
    cdata.load_labels(path)
    loaded = cdata.obs_computed["a"]
    assert loaded.iloc[:10].tolist() == ["x", "y"] * 5 and loaded.iloc[10:].isna().all()
    assert isinstance(loaded.dtype, pd.CategoricalDtype)
    with pytest.raises(KeyError, match="nothing"):
        cdata.load_labels(path, keys="nothing")


def test_labels_of_a_compacted_dataset_come_back_to_the_original_data_in_a_new_session(
    cdata, tmp_path, deltacells_kwargs
):
    # the workflow: slice, compact, cluster the cells of the compacted dataset, and take the labels back
    t_cells = (cdata.obs["cell_type"] == "T cell").to_numpy()
    view = cdata[t_cells]
    compact = _compact(view, tmp_path, deltacells_kwargs)  # what another session opens
    assert compact.cell_ids.tolist() == view.cell_ids.tolist()

    sub = compact.to_anndata()
    sub.obs["subtype"] = pd.Categorical(["s" + str(i % 2) for i in range(sub.n_obs)])
    compact.update_labels_from_anndata(sub, "subtype", as_key="t_cell_subtype")
    path = str(tmp_path / "t_cell_labels.parquet")
    compact.save_labels("t_cell_subtype", path)

    cdata.load_labels(path)
    labels = cdata.obs_computed["t_cell_subtype"]
    assert labels.notna().to_numpy().tolist() == t_cells.tolist()
    assert labels[t_cells].tolist() == sub.obs["subtype"].tolist()
    # the view and the root can both use them to select cells
    assert cdata[labels == "s0"].cell_ids.tolist() == compact.cell_ids[::2].tolist()


def test_loading_labels_of_cells_that_the_data_does_not_have_raises(cdata, tmp_path):
    view = cdata[POSITIONS]
    view.obs_computed["x"] = np.arange(len(POSITIONS)) * 1.0
    path = str(tmp_path / "x.parquet")
    view.save_labels("x", path)
    narrow = cdata[np.arange(10)]
    with pytest.raises(KeyError, match="not among"):
        narrow.load_labels(path)
    assert len(narrow.obs_computed) == 0
    cdata.load_labels(path)
    assert cdata.obs_computed["x"].iloc[POSITIONS].tolist() == (np.arange(len(POSITIONS)) * 1.0).tolist()
