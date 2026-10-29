# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import sys
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellarium.ml.api.preprocessing import highly_variable_genes
from cellarium.ml.api.tools import pca, scvi

ALL_CELLS = np.arange(40)


def counts_of(cdata, positions):
    """The counts of the cells at ``positions``, read from the data."""
    return cdata.datamodule.dadc.read(np.asarray(positions)).X.toarray()


def positions_of(cdata, adata):
    return cdata.obs_names.get_indexer(adata.obs_names)


@pytest.fixture
def trained_pca(cdata):
    highly_variable_genes(cdata, n_top_genes=10, flavor="seurat", accelerator="cpu")
    pca(cdata, n_components=3, zscore=True, accelerator="cpu")


# ---- all the cells ----


def test_all_the_cells_come_with_everything_of_the_data(cdata):
    cdata.hvg = np.arange(30) % 2 == 0
    cdata.obsm["emb"] = np.arange(80, dtype=np.float32).reshape(40, 2)
    cdata.obs_computed["flag"] = np.arange(40) % 3 == 0
    cdata.obs_computed["score"] = np.arange(40) * 1.5

    adata = cdata.to_anndata()

    assert isinstance(adata, ad.AnnData) and adata.shape == (40, 30)
    assert sp.issparse(adata.X) and adata.X.dtype == np.float32
    assert np.array_equal(adata.X.toarray(), counts_of(cdata, ALL_CELLS))
    assert adata.obs_names.tolist() == cdata.obs_names.tolist()
    assert adata.obs["cellarium_cell_id"].tolist() == cdata.cell_ids.tolist()
    assert adata.var_names.tolist() == cdata.var.index.tolist()
    assert adata.obs["cell_type"].tolist() == cdata.obs["cell_type"].tolist()
    assert np.array_equal(adata.obs["n_counts"], cdata.obs["n_counts"])
    assert np.array_equal(adata.var["highly_variable"], np.arange(30) % 2 == 0)
    assert np.array_equal(adata.obsm["emb"], cdata.obsm["emb"])
    assert adata.obs["computed_flag"].tolist() == (np.arange(40) % 3 == 0).tolist()
    assert np.array_equal(adata.obs["computed_score"], np.arange(40) * 1.5)
    assert "computed_flag" not in cdata.obs.columns  # the data is left as it was


def test_without_hvg_no_gene_is_highly_variable(cdata):
    assert not cdata.to_anndata().var["highly_variable"].any()


def test_a_view_gives_its_cells_in_the_order_of_the_data_with_all_the_categories(cdata):
    # no B cell among them: the categories of the columns must still be those of the data, in its order
    keep = np.flatnonzero((cdata.obs["cell_type"] != "B cell").to_numpy())
    assert 0 < len(keep) < 40
    view = cdata[keep[::-1]]

    adata = view.to_anndata()

    assert adata.obs_names.tolist() == cdata.obs_names[keep].tolist()
    assert np.array_equal(adata.X.toarray(), counts_of(cdata, keep))
    categories = cdata.datamodule.dadc.obs_categories("cell_type").tolist()
    assert adata.obs["cell_type"].cat.categories.tolist() == categories
    assert "B cell" not in set(adata.obs["cell_type"])


def test_the_obsm_and_obs_computed_of_a_view_are_those_of_its_cells(cdata):
    cdata.obsm["emb"] = np.arange(80, dtype=np.float32).reshape(40, 2)
    cdata.obs_computed["flag"] = pd.Series(np.arange(40) % 2 == 0, index=cdata.obs.index)
    positions = np.array([3, 8, 21, 39])

    adata = cdata[positions].to_anndata()

    assert np.array_equal(adata.obsm["emb"], cdata.obsm["emb"][positions])
    assert adata.obs["computed_flag"].tolist() == [False, True, False, False]
    assert adata.obs["computed_flag"].index.tolist() == adata.obs_names.tolist()


def test_obs_columns_of_h5ad_files_get_the_categories_of_the_schema_in_its_order(h5ad_cdata):
    from cellarium.ml.api._to_anndata import _with_schema_categories

    dadc = h5ad_cdata.datamodule.dadc
    categories = dadc.obs_categories("cell_type").tolist()
    names = ["NK cell", "T cell", "B cell"]
    reordered = pd.DataFrame({"cell_type": pd.Categorical(names, categories=sorted(names))})
    as_strings = pd.DataFrame({"cell_type": names, "n_counts": [1, 2, 3]})  # e.g. when merging shards loses the dtype

    for obs in (reordered, as_strings):
        column = _with_schema_categories(obs, dadc)["cell_type"]
        assert column.cat.categories.tolist() == categories
        assert column.tolist() == names


# ---- downsampling at random ----


def test_random_downsampling_keeps_the_target_number_of_cells_in_the_order_of_the_data(cdata):
    adata = cdata.to_anndata("random", 12, seed=1)

    positions = positions_of(cdata, adata)
    assert adata.n_obs == 12 and len(set(positions)) == 12
    assert np.all(np.diff(positions) > 0)
    assert np.array_equal(adata.X.toarray(), counts_of(cdata, positions))
    assert adata.obs["cell_type"].tolist() == cdata.obs["cell_type"].to_numpy()[positions].tolist()


def test_random_downsampling_depends_on_the_seed_only(cdata):
    first = cdata.to_anndata("random", 12, seed=1).obs_names.tolist()
    assert cdata.to_anndata("random", 12, seed=1).obs_names.tolist() == first
    assert cdata.to_anndata("random", 12, seed=2).obs_names.tolist() != first
    default = cdata.to_anndata("random", 12)
    assert default.obs_names.tolist() == cdata.to_anndata("random", 12, seed=0).obs_names.tolist()


def test_a_target_of_at_least_the_number_of_cells_keeps_them_all(cdata):
    for method in ("random", "geometric_sketch"):
        adata = cdata.to_anndata(method, 40)
        assert adata.obs_names.tolist() == cdata.obs_names.tolist()
        assert cdata.to_anndata(method, 1000).n_obs == 40


def test_downsampling_a_view(cdata):
    view = cdata[np.arange(5, 35)]
    adata = view.to_anndata("random", 10)
    assert adata.n_obs == 10 and set(adata.obs_names) <= set(view.obs_names)
    assert set(adata.obs["cellarium_cell_id"]) <= set(view.cell_ids)


# ---- geometric sketch ----


def test_a_sketch_without_a_pca_warns_and_uses_a_random_projection(cdata):
    with pytest.warns(UserWarning, match=r"random projection.*cml\.tl\.pca"):
        adata = cdata.to_anndata("geometric_sketch", 10, geometric_sketch_kwargs={"accelerator": "cpu"})

    positions = positions_of(cdata, adata)
    assert adata.n_obs == 10 and np.all(np.diff(positions) > 0)
    assert np.array_equal(adata.X.toarray(), counts_of(cdata, positions))
    assert adata.obs["cell_type"].tolist() == cdata.obs["cell_type"].to_numpy()[positions].tolist()
    assert list(adata.obsm) == []  # nothing is trained, and the random projection is not an embedding to keep
    assert "computed_in_sketch" not in adata.obs  # the mask is left in the data by the sketch, not in the AnnData
    assert cdata.obs_computed["in_sketch"].sum() == 10


def test_a_sketch_uses_the_pca_by_default_and_gives_its_embedding(cdata, trained_pca, monkeypatch):
    module = sys.modules["cellarium.ml.api.tools.geometric_sketch"]
    calls = []
    sketch = module.geometric_sketch

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return sketch(*args, **kwargs)

    monkeypatch.setattr(module, "geometric_sketch", spy)

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*random projection.*")
        adata = cdata.to_anndata("geometric_sketch", 10, seed=5, geometric_sketch_kwargs={"accelerator": "cpu"})

    assert calls[0]["embedding_module_key"] == "pca"
    assert calls[0]["target_n_cells"] == 10 and calls[0]["seed"] == 5 and calls[0]["return_new_adata"]
    assert adata.n_obs == 10
    assert adata.obsm["X_pca"].shape == (10, 3)
    assert not any(key.startswith("X_") and key != "X_pca" for key in adata.obsm)  # not "onepass"

    explicit = {"embedding_module_key": None, "accelerator": "cpu"}
    cdata.to_anndata("geometric_sketch", 10, geometric_sketch_kwargs=explicit)
    assert calls[1]["embedding_module_key"] is None  # given explicitly, so there is no warning either


def test_the_sketch_is_reproducible_given_the_seed(cdata):
    kwargs = {"geometric_sketch_kwargs": {"embedding_module_key": None, "accelerator": "cpu"}, "seed": 7}
    first = cdata.to_anndata("geometric_sketch", 10, **kwargs).obs_names.tolist()
    assert cdata.to_anndata("geometric_sketch", 10, **kwargs).obs_names.tolist() == first


# ---- embeddings ----


def test_embeddings_of_a_view_with_an_assigned_module_are_those_of_the_same_cells_of_the_data(cdata, trained_pca):
    full = cdata.to_anndata()
    assert full.obsm["X_pca"].shape == (40, 3)
    positions = np.array([2, 3, 17, 18, 30])
    view = cdata[positions]
    assert "pca" not in view.trained_modules  # a view starts without the modules of its parent

    assert "X_pca" not in view.to_anndata().obsm
    view.trained_modules["pca"] = cdata.trained_modules["pca"]
    adata = view.to_anndata()

    np.testing.assert_allclose(adata.obsm["X_pca"], full.obsm["X_pca"][positions], rtol=1e-4, atol=1e-4)


def test_scvi_embeds_a_view_with_the_batch_column_it_was_trained_on(cdata):
    scvi(cdata, batch_key="cell_type", n_latent=3, max_epochs=1, accelerator="cpu")
    full = cdata.to_anndata()
    assert full.obsm["X_scvi"].shape == (40, 3)
    # without B cells: the codes of the batch column must still be those of the full data
    positions = np.flatnonzero((cdata.obs["cell_type"] != "B cell").to_numpy())
    view = cdata[positions]
    view.trained_modules["scvi"] = cdata.trained_modules["scvi"]

    adata = view.to_anndata()

    np.testing.assert_allclose(adata.obsm["X_scvi"], full.obsm["X_scvi"][positions], rtol=1e-4, atol=1e-4)
    assert "batch_index_n" not in view.datamodule.batch_keys  # the datamodule is put back


def test_collect_embeddings_chooses_the_modules(cdata, trained_pca):
    scvi(cdata, batch_key="cell_type", n_latent=2, max_epochs=1, accelerator="cpu")
    assert {"pca", "scvi", "onepass"} <= set(cdata.trained_modules)

    assert {k for k in cdata.to_anndata().obsm} == {"X_pca", "X_scvi"}  # `onepass` is not an embedding
    assert list(cdata.to_anndata(collect_embeddings=False).obsm) == []
    assert list(cdata.to_anndata(collect_embeddings=["scvi"]).obsm) == ["X_scvi"]
    assert list(cdata.to_anndata(collect_embeddings="pca").obsm) == ["X_pca"]
    assert cdata.to_anndata(collect_embeddings=[]).obsm.keys() == set()


def test_collect_embeddings_rejects_modules_that_are_not_trained_embeddings(cdata, trained_pca):
    with pytest.raises(ValueError, match=r"no trained module 'scvi'.*Available"):
        cdata.to_anndata(collect_embeddings=["scvi"])
    with pytest.raises(ValueError, match="not one of the embedding models"):
        cdata.to_anndata(collect_embeddings=["onepass"])


def test_an_embedding_that_fails_is_skipped_with_a_warning_unless_it_was_asked_for(cdata, trained_pca, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("no luck")

    monkeypatch.setattr("cellarium.ml.api._to_anndata._embed", broken)

    with pytest.warns(UserWarning, match="'pca' could not be computed.*no luck"):
        adata = cdata.to_anndata()
    assert "X_pca" not in adata.obsm
    with pytest.raises(RuntimeError, match="no luck"):
        cdata.to_anndata(collect_embeddings=["pca"])


def test_an_obsm_entry_is_not_replaced_by_an_embedding(cdata, trained_pca):
    cdata.obsm["X_pca"] = np.zeros((40, 2))

    with pytest.warns(UserWarning, match="obsm\\['X_pca'\\] already exists"):
        adata = cdata.to_anndata()
    assert adata.obsm["X_pca"].shape == (40, 2)
    with pytest.raises(ValueError, match="already exists"):
        cdata.to_anndata(collect_embeddings=["pca"])


# ---- arguments ----


def test_a_target_needs_a_method_and_a_method_needs_a_target(cdata):
    with pytest.raises(ValueError, match="needs a downsample_method"):
        cdata.to_anndata(target_n_cells=10)
    with pytest.raises(ValueError, match="needs target_n_cells"):
        cdata.to_anndata("random")
    with pytest.raises(ValueError, match="needs target_n_cells"):
        cdata.to_anndata("geometric_sketch")


def test_the_method_and_target_must_be_valid(cdata):
    with pytest.raises(ValueError, match="must be None, 'random' or 'geometric_sketch'"):
        cdata.to_anndata("sketch", 10)  # type: ignore[arg-type]
    for target in (0, -3, 2.5):
        with pytest.raises(ValueError, match="positive integer"):
            cdata.to_anndata("random", target)  # type: ignore[arg-type]


def test_geometric_sketch_kwargs_only_go_with_a_sketch_and_do_not_set_what_the_method_sets(cdata):
    with pytest.raises(ValueError, match="only apply to downsample_method='geometric_sketch'"):
        cdata.to_anndata("random", 10, geometric_sketch_kwargs={"n_pcs": 2})
    for name in ("target_n_cells", "seed", "return_new_adata"):
        with pytest.raises(ValueError, match=name):
            cdata.to_anndata("geometric_sketch", 10, geometric_sketch_kwargs={name: 1})


def test_cells_that_do_not_fit_are_not_read(cdata):
    with pytest.raises(MemoryError):
        cdata.to_anndata(max_gb=1e-9)
    with pytest.raises(MemoryError):
        cdata.to_anndata("random", 20, max_gb=1e-9)
