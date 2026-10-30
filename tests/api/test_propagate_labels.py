# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import inspect
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from cellarium.ml.api.preprocessing import highly_variable_genes
from cellarium.ml.api.tools import embed, pca, propagate_labels

module = sys.modules["cellarium.ml.api.tools.propagate_labels"]  # (the package's name is the function)

needs_pynndescent = pytest.mark.skipif(importlib.util.find_spec("pynndescent") is None, reason="needs pynndescent")

# the `cdata` fixture (tests/api/conftest.py): 40 cells in two h5ad files of 20, as h5ad files or as a deltacells
# dataset in tiles of 15, with obs columns `cell_type` and `n_counts`

CENTERS = {"T cell": [0.0, 0.0], "B cell": [10.0, 0.0], "NK cell": [0.0, 10.0]}
UNLABELED = np.array([2, 9, 17, 23, 31, 38])


def _set_up(cdata, embedding_key="emb", labels_key="cl"):
    """An embedding in which the cells are in clumps by cell type, and labels (the cell types) of all
    but a few cells."""
    types = cdata.obs["cell_type"].astype(str).to_numpy()
    noise = np.random.default_rng(0).normal(0, 0.3, size=(40, 2))
    cdata.obsm[embedding_key] = (np.array([CENTERS[t] for t in types]) + noise).astype(np.float32)
    labels = pd.Series(types.copy(), dtype="string")  # a copy: the Series would share the array that gets NAs
    labels[UNLABELED] = pd.NA
    cdata.obs_computed[labels_key] = labels
    return types


def _propagate(cdata, labels_key, rep, **kwargs):
    kwargs.setdefault("k", 5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return propagate_labels(cdata, labels_key, rep, **kwargs)


def _warnings_of_propagate(cdata, labels_key, rep, **kwargs) -> list[str]:
    kwargs.setdefault("k", 5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        propagate_labels(cdata, labels_key, rep, **kwargs)
    return [str(w.message) for w in caught if "does not reproduce the labels" in str(w.message)]


def test_the_unlabeled_cells_get_the_label_of_their_neighbors_and_the_labeled_ones_keep_theirs(cdata):
    types = _set_up(cdata)
    result = _propagate(cdata, "cl", "emb")

    labels = cdata.obs_computed["cl_propagated"]
    assert isinstance(labels.dtype, pd.CategoricalDtype) and labels.index.equals(cdata.cell_ids)
    assert labels.tolist() == types.tolist()  # all of them: the clumps are separated
    assert labels.notna().all()

    assert result.key_added == "cl_propagated" and result.rep == "emb"
    assert result.n_reference == 34 and result.n_propagated == 6
    assert result.agreement == 1.0 and result.n_checked == 34
    assert sorted(result.per_label.index) == sorted(CENTERS)
    assert (result.per_label["agreement"] == 1.0).all() and result.per_label["n_reference"].sum() == 34
    assert result.per_label["most_confused_with"].isna().all()

    # the original labels are left as they were
    assert cdata.obs_computed["cl"].isna().to_numpy().nonzero()[0].tolist() == UNLABELED.tolist()


def test_confidence_and_distance_are_stored_for_every_cell(cdata):
    _set_up(cdata)
    _propagate(cdata, "cl", "emb")

    confidence = cdata.obs_computed["cl_propagated_confidence"]
    distance = cdata.obs_computed["cl_propagated_distance"]
    assert confidence.index.equals(cdata.cell_ids) and distance.index.equals(cdata.cell_ids)
    labeled = np.setdiff1d(np.arange(40), UNLABELED)
    assert (confidence.to_numpy()[labeled] == 1.0).all() and (distance.to_numpy()[labeled] == 0.0).all()
    assert (confidence.to_numpy()[UNLABELED] > 0.99).all()  # all the neighbors are of the same clump
    assert (distance.to_numpy()[UNLABELED] > 0).all() and (distance.to_numpy()[UNLABELED] < 3).all()
    # so that cells can be left out by slicing on them
    assert cdata[confidence > 0.5].n_obs == 40
    assert cdata[distance > 0].cell_ids.tolist() == UNLABELED.tolist()


def test_a_vote_that_is_split_has_a_lower_confidence(cdata):
    types = _set_up(cdata)
    # a cell halfway between the T cells and the B cells
    cdata.obsm["emb"][UNLABELED[0]] = [5.0, 0.0]
    _propagate(cdata, "cl", "emb", k=8)
    confidence = cdata.obs_computed["cl_propagated_confidence"].to_numpy()
    assert confidence[UNLABELED[0]] < 0.8  # the T cells and the B cells split the vote
    assert (confidence[UNLABELED[1:]] > 0.95).all()
    assert types is not None


def test_the_names_of_the_results_and_the_settings(cdata):
    _set_up(cdata)
    result = _propagate(cdata, "cl", "emb", key_added="final", weights="uniform", chunk_size=4, n_check=10)
    assert result.key_added == "final" and result.n_checked == 10
    assert {"final", "final_confidence", "final_distance"} <= set(cdata.obs_computed)
    assert cdata.obs_computed["final"].notna().all()


def test_the_labels_of_a_view_are_propagated_within_the_view(cdata):
    types = _set_up(cdata)
    view = cdata[np.arange(5, 40)]  # inherits the embedding and the labels
    result = _propagate(view, "cl", "emb")
    assert result.n_propagated == int((UNLABELED >= 5).sum())
    assert view.obs_computed["cl_propagated"].tolist() == types[5:].tolist()
    assert "cl_propagated" not in cdata.obs_computed
    # and the labels of the view can be brought to the data it was taken from
    cdata.obs_computed.update_from(view, "cl_propagated")
    assert cdata.obs_computed["cl_propagated"].iloc[5:].tolist() == types[5:].tolist()
    assert cdata.obs_computed["cl_propagated"].iloc[:5].isna().all()


def test_a_rep_that_is_a_trained_module_is_embedded_first(cdata):
    highly_variable_genes(cdata, n_top_genes=10, flavor="seurat", accelerator="cpu")
    pca(cdata, n_components=3, zscore=True, accelerator="cpu")
    _set_up(cdata)
    result = _propagate(cdata, "cl", "pca", accelerator="cpu")
    assert result.rep == "X_pca"
    assert cdata.obsm["X_pca"].shape == (40, 3)
    embedded = cdata.obsm["X_pca"]
    _propagate(cdata, "cl", "pca", accelerator="cpu")
    assert cdata.obsm["X_pca"] is embedded  # not embedded again


def _same_results(cdata, one, other):
    np.testing.assert_array_equal(
        cdata.obs_computed[one].to_numpy().astype(str), cdata.obs_computed[other].to_numpy().astype(str)
    )
    for suffix in ("_confidence", "_distance"):
        np.testing.assert_allclose(
            cdata.obs_computed[one + suffix], cdata.obs_computed[other + suffix], rtol=1e-3, atol=1e-4
        )


@needs_pynndescent
def test_the_threads_give_the_same_labels_as_one_thread(cdata):
    import numba

    _set_up(cdata)
    _propagate(cdata, "cl", "emb", key_added="serial", method="approximate", n_jobs=1)
    assert numba.get_num_threads() == numba.config.NUMBA_NUM_THREADS  # restored
    _propagate(cdata, "cl", "emb", key_added="parallel", method="approximate", n_jobs=2, chunk_size=2)
    assert numba.get_num_threads() == numba.config.NUMBA_NUM_THREADS
    assert cdata.obs_computed["parallel"].notna().all()
    _same_results(cdata, "serial", "parallel")


@needs_pynndescent
def test_the_exact_and_the_approximate_search_give_the_same_labels(cdata):
    types = _set_up(cdata)
    exact = _propagate(cdata, "cl", "emb", key_added="exact", method="exact", accelerator="cpu")
    approximate = _propagate(cdata, "cl", "emb", key_added="approximate", method="approximate")
    assert (exact.method, approximate.method) == ("exact", "approximate")
    assert cdata.obs_computed["exact"].tolist() == types.tolist()
    assert exact.agreement == approximate.agreement == 1.0
    _same_results(cdata, "exact", "approximate")


def test_auto_chooses_the_exact_search_for_a_small_search_and_the_approximate_one_for_a_large_one(cdata, monkeypatch):
    _set_up(cdata)
    assert _propagate(cdata, "cl", "emb", accelerator="cpu").method == "exact"

    approximate_indexes = []

    def approximate_index(x, k, seed):  # an exact index stands in, so that pynndescent is not needed
        approximate_indexes.append(x)
        return module._ExactIndex(x, torch.device("cpu"))

    monkeypatch.setattr(module, "_approximate_index", approximate_index)
    monkeypatch.setitem(sys.modules, "pynndescent", object())
    pairs = (6 + 34) * 34  # the cells to label and to check, times the labeled cells
    monkeypatch.setattr(module, "_MAX_EXACT_PAIRS", {"cpu": pairs - 1, "mps": 0, "cuda": 0})
    assert _propagate(cdata, "cl", "emb", accelerator="cpu").method == "approximate"
    assert len(approximate_indexes) == 1
    monkeypatch.setattr(module, "_MAX_EXACT_PAIRS", {"cpu": pairs, "mps": 0, "cuda": 0})
    assert _propagate(cdata, "cl", "emb", accelerator="cpu").method == "exact"
    assert len(approximate_indexes) == 1


def test_the_exact_search_does_not_need_pynndescent(cdata, monkeypatch):
    types = _set_up(cdata)
    monkeypatch.setitem(sys.modules, "pynndescent", None)  # an import of it fails
    _propagate(cdata, "cl", "emb", method="exact", accelerator="cpu")
    assert cdata.obs_computed["cl_propagated"].tolist() == types.tolist()
    with pytest.raises(ImportError, match="method='exact'"):
        propagate_labels(cdata, "cl", "emb", method="approximate")


def test_the_exact_neighbors_are_the_nearest_ones_whatever_the_tiles_and_the_offset(monkeypatch):
    monkeypatch.setattr(module, "_REFERENCE_TILE", 7)  # the 50 points are in 8 tiles, the last with 1 point
    monkeypatch.setattr(module, "_QUERY_TILE", 8)
    rng = np.random.default_rng(0)
    points = (rng.normal(size=(50, 4)) + 1000.0).astype(np.float32)  # far from the origin: products would cancel
    points[10] = points[3]  # a duplicate
    queries = (rng.normal(size=(20, 4)) + 1000.0).astype(np.float32)
    queries[0] = points[3]

    indices, distances = module._ExactIndex(points, torch.device("cpu")).query(queries, k=5)

    truth = np.sqrt(((queries[:, None].astype(np.float64) - points[None].astype(np.float64)) ** 2).sum(-1))
    order = np.argsort(truth, axis=1, kind="stable")[:, :5]
    np.testing.assert_allclose(distances, np.take_along_axis(truth, order, 1), atol=1e-4)
    assert (np.sort(indices[1:], axis=1) == np.sort(order[1:], axis=1)).all()
    assert set(indices[0, :2]) == {3, 10} and (distances[0, :2] == 0).all()  # the duplicates
    assert (np.diff(distances, axis=1) >= 0).all()
    # fewer points than k: all of them
    assert module._ExactIndex(points[:3], torch.device("cpu")).query(queries, k=5)[0].shape == (20, 3)


def test_the_check_holds_out_at_most_20_000_cells_by_default():
    assert inspect.signature(propagate_labels).parameters["n_check"].default == 20_000


def test_an_embedding_on_disk_is_read_in_chunks(cdata):
    types = _set_up(cdata)
    cdata.max_gb = 1e-9
    highly_variable_genes(cdata, n_top_genes=10, flavor="seurat", accelerator="cpu")
    pca(cdata, n_components=3, zscore=True, accelerator="cpu")
    embed(cdata, "pca", key_added="on_disk", accelerator="cpu")
    assert isinstance(cdata.obsm["on_disk"], np.memmap)
    _propagate(cdata, "cl", "on_disk", chunk_size=2)
    assert cdata.obs_computed["cl_propagated"].notna().all()
    assert types is not None


# ---- the check of the labels against the embedding ----


def test_an_embedding_that_does_not_reproduce_the_labels_is_warned_about_but_used(cdata):
    types = _set_up(cdata)
    # labels that have nothing to do with the clumps
    shuffled = pd.Series(np.random.default_rng(1).permutation(types).copy(), dtype="string")
    shuffled[UNLABELED] = pd.NA
    cdata.obs_computed["cl"] = shuffled

    messages = _warnings_of_propagate(cdata, "cl", "emb")
    assert len(messages) == 1
    assert "held-out" in messages[0] and "different space" in messages[0] and "'emb'" in messages[0]
    assert cdata.obs_computed["cl_propagated"].notna().all()  # the labels are given anyway


def test_the_check_reports_which_labels_are_not_reproduced(cdata):
    types = _set_up(cdata)
    # two clumps given the same labels in the embedding's eyes: B cells are labeled half "B1", half "B2"
    labels = pd.Series(types.copy(), dtype="string")
    b_cells = np.flatnonzero(types == "B cell")
    labels[b_cells[::2]] = "B1"
    labels[b_cells[1::2]] = "B2"
    labels[UNLABELED] = pd.NA
    cdata.obs_computed["cl"] = labels

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = propagate_labels(cdata, "cl", "emb", k=5)
    table = result.per_label
    assert table.loc["T cell", "agreement"] == 1.0
    assert table.loc["B1", "agreement"] < 0.8 and table.loc["B2", "agreement"] < 0.8
    assert table.loc["B1", "most_confused_with"] == "B2" or table.loc["B2", "most_confused_with"] == "B1"
    assert 0.7 < result.agreement < 1.0


def test_no_warning_when_the_labels_are_reproduced(cdata):
    _set_up(cdata)
    assert _warnings_of_propagate(cdata, "cl", "emb") == []


# ---- arguments ----


def test_propagate_labels_checks_what_it_is_given(cdata):
    _set_up(cdata)
    with pytest.raises(KeyError, match="No 'nothing'"):
        propagate_labels(cdata, "nothing", "emb")
    with pytest.raises(KeyError, match="neither a key of obsm"):
        propagate_labels(cdata, "cl", "nothing")
    with pytest.raises(ValueError, match="weights"):
        propagate_labels(cdata, "cl", "emb", weights="other")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="method"):
        propagate_labels(cdata, "cl", "emb", method="other")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_jobs"):
        propagate_labels(cdata, "cl", "emb", n_jobs=0)
    with pytest.raises(ValueError, match="at least k"):
        propagate_labels(cdata, "cl", "emb", k=40)
    cdata.obs_computed["full"] = cdata.obs["cell_type"].astype(str).to_numpy()
    with pytest.raises(ValueError, match="nothing to propagate"):
        propagate_labels(cdata, "full", "emb", k=5)
    cdata.obsm["bad"] = np.zeros((40, 2, 2))
    with pytest.raises(ValueError, match="one row for each"):
        propagate_labels(cdata, "cl", "bad", k=5)
    assert "cl_propagated" not in cdata.obs_computed
