# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from cellarium.ml.api.cellariumdata import CellariumData
from cellarium.ml.api.tools import embed, scanvi, scanvi_annotate, scvi
from cellarium.ml.api.tools.scanvi import _fill_unlabeled, _label_counts, _node_depths, _select_nodes
from cellarium.ml.utilities.core import call_func_with_batch

CL_NAMES = ["R", "T", "B", "Mono", "CD4", "CD8", "unknown"]  # R -> {T -> {CD4, CD8}, B, Mono}, and a stray "unknown"
ONTOLOGY_KWARGS = {"frontier_min_cells": 3}  # the toy data has few cells


def _toy_ontology() -> tuple[torch.Tensor, list[str]]:
    index = {name: i for i, name in enumerate(CL_NAMES)}
    descendants = torch.eye(len(CL_NAMES))
    for ancestor, below in {"R": ["T", "B", "Mono", "CD4", "CD8"], "T": ["CD4", "CD8"]}.items():
        for name in below:
            descendants[index[ancestor], index[name]] = 1.0
    return descendants, list(CL_NAMES)


@pytest.fixture
def cdata(tmp_path) -> CellariumData:
    """40 cells in two h5ad files with a batch, flat cell types and ontology labels,
    some unlabeled (NaN or "unknown")."""
    rng = np.random.default_rng(0)
    n_genes = 30
    paths = []
    for i in range(2):
        j = np.arange(20)
        flat = np.array(["T", "B", "NK"], dtype=object)[j % 3]
        ontology = np.array(["CD8", "B", "Mono", "T", "CD4", "CD4"], dtype=object)[j % 6]
        for labels in (flat, ontology):
            labels[j % 7 == 0] = "unknown"
            labels[j % 5 == 0] = None
        few = np.full(20, None, dtype=object)
        few[3] = ["T", "B"][i]
        obs = pd.DataFrame(
            {
                "batch": pd.Categorical(np.where(j % 2, "batch1", "batch0"), categories=["batch0", "batch1"]),
                "cell_type": pd.Categorical(flat, categories=["T", "B", "NK", "unknown"]),
                "cl": pd.Categorical(ontology, categories=["CD4", "CD8", "B", "Mono", "T", "unknown"]),
                # two labeled cells in all, and none
                "few": pd.Categorical(few, categories=["T", "B"]),
                "none": pd.Categorical([None] * 20, categories=["x"]),
            },
            index=[f"file{i}_cell{k}" for k in j],
        )
        obs.index.name = "barcode"
        x = sp.csr_matrix(rng.poisson(lam=3.0, size=(20, n_genes)).astype(np.float32))
        adata = ad.AnnData(X=x, obs=obs, var=pd.DataFrame(index=[f"gene{k}" for k in range(n_genes)]))
        path = tmp_path / f"scanvi_{i}.h5ad"
        adata.write_h5ad(path)
        paths.append(str(path))
    return CellariumData(h5ad_paths=paths)


# ---- the helpers ----


@pytest.mark.parametrize("dtype", ["category", "object", "string"])
def test_missing_labels_are_filled_with_the_unlabeled_category(dtype):
    labels = pd.Series(["a", None, "b", None], dtype=dtype)
    filled = _fill_unlabeled(labels, "unknown")
    assert filled.dtype == object and filled.tolist() == ["a", "unknown", "b", "unknown"]


def test_label_counts_counts_missing_values_as_unlabeled(cdata):
    counts = _label_counts(cdata, "cell_type", "unknown")
    # j % 5 == 0 is missing (4 of 20), the other j % 7 == 0 (7, 14) are "unknown", in each of the two files
    assert counts["unknown"] == 2 * (4 + 2)
    assert counts.sum() == 40 and set(counts.index) == {"T", "B", "NK", "unknown"}


def test_node_depths_are_the_longest_paths_from_the_root():
    # R -> A -> B -> D and R -> C -> D (D is reached by two paths, of lengths 3 and 2), and a separate root E
    names = ["R", "A", "B", "C", "D", "E"]
    index = {name: i for i, name in enumerate(names)}
    descendants = torch.eye(len(names))
    for ancestor, below in {"R": "ABCD", "A": "BD", "B": "D", "C": "D"}.items():
        for name in below:
            descendants[index[ancestor], index[name]] = 1.0
    assert _node_depths(descendants).tolist() == [0, 1, 2, 1, 3, 0]


def test_select_nodes_takes_the_deepest_node_that_passes_the_threshold():
    depth = torch.tensor([0, 1, 1, 2, 2])  # R, T, B, CD4, CD8
    probs = torch.tensor(
        [
            [1.0, 0.9, 0.1, 0.5, 0.4],  # CD4 and CD8 both pass at depth 2: the likelier one
            [1.0, 0.9, 0.1, 0.1, 0.1],  # the deepest to pass is T
            [1.0, 0.1, 0.1, 0.05, 0.05],  # only R passes
            [0.1, 0.1, 0.1, 0.05, 0.05],  # none
        ]
    )
    index, confidence, assigned = _select_nodes(probs, depth, threshold=0.2)
    assert index[:3].tolist() == [3, 1, 0] and assigned.tolist() == [True, True, True, False]
    assert confidence.tolist() == pytest.approx([0.5, 0.9, 1.0, 0.0])


def test_select_nodes_without_depth_is_the_most_likely_class_if_it_passes():
    probs = torch.tensor([[0.2, 0.7, 0.1], [0.3, 0.3, 0.4]])
    index, confidence, assigned = _select_nodes(probs, torch.zeros(3, dtype=torch.long), threshold=0.5)
    assert assigned.tolist() == [True, False] and index[0].item() == 1
    assert confidence.tolist() == pytest.approx([0.7, 0.0])


# ---- flat ----


def test_flat_scanvi_trains_without_the_unlabeled_category_as_a_class(cdata):
    scanvi(cdata, batch_key="batch", labels_key="cell_type", n_latent=3, max_epochs=1, accelerator="cpu")
    trained = cdata.trained_modules["scanvi"]
    model = trained.module.model
    assert model.classifier_type == "flat" and model.class_names == ["B", "NK", "T"]
    assert trained.config["n_unlabeled"] == 12 and trained.config["n_labeled"] == 28
    # only the batch is needed to run it, and the datamodule is put back
    assert list(trained.batch_keys) == ["batch_index_n"]
    assert not {"batch_index_n", "cell_type_labels_n"} & set(cdata.datamodule.batch_keys)


def test_scanvi_prints_the_labeled_and_unlabeled_cells(cdata, capsys):
    scanvi(cdata, batch_key="batch", labels_key="cell_type", n_latent=3, max_epochs=1, accelerator="cpu")
    assert "28 labeled and 12 unlabeled cells (30.0% unlabeled), with 3 cell types" in capsys.readouterr().out


def test_scanvi_warns_when_almost_all_of_the_cells_are_unlabeled(cdata):
    with pytest.warns(UserWarning, match="95.0% of the cells are unlabeled"):
        scanvi(cdata, batch_key="batch", labels_key="few", n_latent=3, max_epochs=1, accelerator="cpu")
    assert "scanvi" in cdata.trained_modules  # and it trains regardless


def test_scanvi_then_embed_and_annotate_a_flat_classifier(cdata):
    scanvi(cdata, batch_key="batch", labels_key="cell_type", n_latent=3, max_epochs=1, accelerator="cpu")
    embed(cdata, "scanvi", accelerator="cpu")
    assert cdata.obsm["X_scanvi"].shape == (40, 3)

    scanvi_annotate(cdata, "scanvi", threshold=0.0, accelerator="cpu", batch_size=7)
    labels = cdata.obs_computed["scanvi_annotation"]
    confidence = cdata.obs_computed["scanvi_annotation_confidence"]
    assert isinstance(labels.dtype, pd.CategoricalDtype) and labels.index.equals(cdata.cell_ids)
    assert set(labels.cat.categories) <= {"B", "NK", "T"}  # never "unknown": the threshold is 0
    assert ((confidence > 0) & (confidence <= 1)).all()

    # with a threshold at 0 the label is the most likely class
    module = cdata.trained_modules["scanvi"].module
    probs = np.zeros((40, 3), dtype=np.float32)
    with cdata.using("scanvi"), torch.no_grad():
        for batch in cdata.predict_batches(module):
            prediction = call_func_with_batch(module.model.predict, batch)
            probs[batch["row_indices_n"].numpy()] = prediction["y_probs_nc"].numpy()
    np.testing.assert_array_equal(labels.astype(str).to_numpy(), np.array(["B", "NK", "T"])[probs.argmax(axis=1)])
    np.testing.assert_allclose(confidence.to_numpy(), probs.max(axis=1), rtol=1e-5)


def test_annotate_gives_unknown_to_the_cells_below_the_threshold(cdata):
    scanvi(cdata, batch_key="batch", labels_key="cell_type", n_latent=3, max_epochs=1, accelerator="cpu")
    scanvi_annotate(cdata, "scanvi", key_added="strict", threshold=1.0, accelerator="cpu")
    assert (cdata.obs_computed["strict"].astype(str) == "unknown").all()
    assert (cdata.obs_computed["strict_confidence"] == 0).all()


# ---- ontology ----


def test_ontology_scanvi_with_a_prepared_ontology_trains_and_annotates(cdata, capsys):
    scanvi(
        cdata,
        batch_key="batch",
        labels_key="cl",
        n_latent=3,
        ontology=_toy_ontology(),
        scanvi_kwargs=ONTOLOGY_KWARGS,
        max_epochs=1,
        accelerator="cpu",
    )
    trained = cdata.trained_modules["scanvi"]
    model = trained.module.model
    assert model.classifier_type == "ontology" and set(model.frontier_cl_names) == {"CD4", "CD8", "B", "Mono"}
    assert "unknown" not in model.active_cl_names  # the unlabeled category is never a target
    assert trained.config["frontier_cl_names"] == model.frontier_cl_names and "ontology" not in trained.config
    assert "a frontier of 4 cell types (6 ontology nodes in all)" in capsys.readouterr().out

    # a threshold of 0 passes every node, so each cell is given a node of the greatest depth: a leaf of the tree
    scanvi_annotate(cdata, "scanvi", threshold=0.0, accelerator="cpu")
    assert set(cdata.obs_computed["scanvi_annotation"].cat.categories) <= {"CD4", "CD8", "B", "Mono"}
    assert (cdata.obs_computed["scanvi_annotation_confidence"] > 0).all()


def test_ontology_from_an_owl_file_is_parsed_into_the_ontology(cdata, monkeypatch):
    module = sys.modules["cellarium.ml.api.tools.scanvi"]  # (the package's name is the function)
    descendants, names = _toy_ontology()
    monkeypatch.setattr(module, "get_cl_descendant_tensor_from_owl", lambda uri: descendants)
    monkeypatch.setattr(module, "get_cl_names_from_owl", lambda uri: names)
    scanvi(
        cdata,
        batch_key="batch",
        labels_key="cl",
        n_latent=3,
        ontology="https://example.org/cl.owl",
        scanvi_kwargs=ONTOLOGY_KWARGS,
        max_epochs=1,
        accelerator="cpu",
    )
    assert cdata.trained_modules["scanvi"].config["ontology"] == "https://example.org/cl.owl"


def test_labels_that_are_not_in_the_ontology_are_an_error(cdata):
    descendants, names = _toy_ontology()
    with pytest.raises(ValueError, match=r"not cell types of the ontology.*'NK'"):
        scanvi(
            cdata,
            batch_key="batch",
            labels_key="cell_type",  # T and B are in the toy ontology, NK is not
            ontology=(descendants, names),
            scanvi_kwargs=ONTOLOGY_KWARGS,
            max_epochs=1,
            accelerator="cpu",
        )
    assert "scanvi" not in cdata.trained_modules


# ---- errors ----


def test_no_labeled_cell_is_an_error(cdata):
    with pytest.raises(ValueError, match="No cell has a label in obs\\['none'\\]"):
        scanvi(cdata, batch_key="batch", labels_key="none", max_epochs=1, accelerator="cpu")
    assert "scanvi" not in cdata.trained_modules


def test_labels_and_batch_must_be_obs_columns(cdata):
    with pytest.raises(ValueError, match="labels_key 'missing' is not a column of obs"):
        scanvi(cdata, batch_key="batch", labels_key="missing", max_epochs=1, accelerator="cpu")
    with pytest.raises(ValueError, match="batch_key 'missing' is not a column of obs"):
        scanvi(cdata, batch_key="missing", labels_key="cl", max_epochs=1, accelerator="cpu")


def test_the_arguments_that_scanvi_sets_cannot_be_overridden(cdata):
    with pytest.raises(ValueError, match="cannot contain.*classifier_type"):
        scanvi(
            cdata,
            batch_key="batch",
            labels_key="cell_type",
            scanvi_kwargs={"classifier_type": "ontology"},
            max_epochs=1,
            accelerator="cpu",
        )


def test_annotate_needs_a_scanvi_module(cdata):
    scvi(cdata, batch_key="batch", n_latent=3, max_epochs=1, accelerator="cpu")
    with pytest.raises(TypeError, match="not an scANVI module"):
        scanvi_annotate(cdata, "scvi", accelerator="cpu")
    assert "scvi_annotation" not in cdata.obs_computed


def test_annotate_needs_a_threshold_between_0_and_1(cdata):
    with pytest.raises(ValueError, match="between 0 and 1"):
        scanvi_annotate(cdata, "scanvi", threshold=1.5, accelerator="cpu")
