# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import numpy as np
import pytest

from cellarium.ml.api.preprocessing import highly_variable_genes
from cellarium.ml.api.tools import embed, geometric_sketch, pca, scvi
from cellarium.ml.api.tools.embed import ensure_embedding

# the `cdata` fixture (tests/api/conftest.py): 40 cells in two h5ad files of 20, as h5ad files or as a deltacells
# dataset in tiles of 15, with obs columns `cell_type` and `n_counts`


@pytest.fixture
def trained_pca(cdata):
    highly_variable_genes(cdata, n_top_genes=10, flavor="seurat", accelerator="cpu")
    pca(cdata, n_components=3, zscore=True, accelerator="cpu")


def test_embed_stores_the_embedding_of_every_cell_in_the_order_of_the_cells(cdata, trained_pca):
    expected = cdata.to_anndata().obsm["X_pca"]  # computed for the cells in memory
    embed(cdata, "pca", accelerator="cpu")

    stored = cdata.obsm["X_pca"]
    assert stored.shape == (40, 3) and stored.dtype == np.float32
    np.testing.assert_allclose(stored, expected, rtol=1e-4, atol=1e-4)


def test_embed_stores_under_the_key_it_is_given(cdata, trained_pca):
    embed(cdata, "pca", key_added="pca_all", accelerator="cpu", batch_size=7)
    assert list(cdata.obsm) == ["pca_all"]
    np.testing.assert_allclose(cdata.obsm["pca_all"], cdata.to_anndata().obsm["X_pca"], rtol=1e-4, atol=1e-4)


def test_embed_goes_to_a_file_when_it_does_not_fit_and_a_view_reads_its_rows_from_there(cdata, trained_pca):
    cdata.max_gb = 1e-9  # nothing fits in memory
    embed(cdata, "pca", accelerator="cpu")
    assert isinstance(cdata.obsm["X_pca"], np.memmap)

    positions = np.array([3, 8, 21, 39])
    view = cdata[positions]
    np.testing.assert_array_equal(np.asarray(view.obsm["X_pca"]), np.asarray(cdata.obsm["X_pca"])[positions])


def test_embed_a_view_embeds_its_cells_with_a_module_assigned_to_it(cdata, trained_pca):
    positions = np.array([1, 4, 16, 22, 30, 39])
    view = cdata[positions]
    with pytest.raises(ValueError, match="No trained module 'pca'"):
        embed(view, "pca", accelerator="cpu")
    view.trained_modules["pca"] = cdata.trained_modules["pca"]
    embed(view, "pca", accelerator="cpu")
    embed(cdata, "pca", accelerator="cpu")
    np.testing.assert_allclose(view.obsm["X_pca"], cdata.obsm["X_pca"][positions], rtol=1e-4, atol=1e-4)


def test_embed_with_scvi_provides_the_batch_column(cdata):
    scvi(cdata, batch_key="cell_type", n_latent=3, max_epochs=1, accelerator="cpu")
    embed(cdata, "scvi", accelerator="cpu")
    assert cdata.obsm["X_scvi"].shape == (40, 3)
    assert "batch_index_n" not in cdata.datamodule.batch_keys  # the datamodule is put back


def test_embed_rejects_a_module_that_cannot_embed(cdata):
    highly_variable_genes(cdata, n_top_genes=10, flavor="seurat", accelerator="cpu")
    pca(cdata, n_components=3, zscore=True, accelerator="cpu")
    assert "onepass" in cdata.trained_modules  # the statistics of the z-score: not an embedding
    with pytest.raises(TypeError, match="cannot predict"):
        embed(cdata, "onepass", accelerator="cpu")
    assert "X_onepass" not in cdata.obsm


# ---- the cached embedding and to_anndata ----


def test_to_anndata_uses_the_embedding_that_embed_made_without_a_warning(cdata, trained_pca):
    embed(cdata, "pca", accelerator="cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        adata = cdata.to_anndata()
    np.testing.assert_array_equal(adata.obsm["X_pca"], cdata.obsm["X_pca"])


def test_a_view_with_a_new_module_does_not_get_the_stale_embedding_of_its_parent(cdata, trained_pca):
    embed(cdata, "pca", accelerator="cpu")
    view = cdata[np.arange(5, 35)]
    # the view inherits the parent's embedding (with the module that made it, if that module is assigned) ...
    view.trained_modules["pca"] = cdata.trained_modules["pca"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        same = view.to_anndata()
    np.testing.assert_array_equal(same.obsm["X_pca"], cdata.obsm["X_pca"][5:35])

    # ... but if the view gets a module of its own, the embedding in the AnnData is that of the new module
    highly_variable_genes(view, n_top_genes=10, flavor="seurat", accelerator="cpu")
    pca(view, n_components=3, zscore=True, accelerator="cpu")
    with pytest.warns(UserWarning, match="earlier module"):
        adata = view.to_anndata()
    assert not np.allclose(adata.obsm["X_pca"], cdata.obsm["X_pca"][5:35], atol=1e-3)
    assert adata.obsm["X_pca"].shape == (30, 3)


def test_an_obsm_entry_that_embed_did_not_make_is_still_not_replaced(cdata, trained_pca):
    cdata.obsm["X_pca"] = np.zeros((40, 2))
    with pytest.warns(UserWarning, match="already exists"):
        adata = cdata.to_anndata()
    assert adata.obsm["X_pca"].shape == (40, 2)


# ---- ensure_embedding (what rep names) ----


def test_a_rep_is_an_obsm_key_or_the_key_of_a_module(cdata, trained_pca):
    cdata.obsm["mine"] = np.zeros((40, 2))
    assert ensure_embedding(cdata, "mine") == "mine"  # used as it is
    assert "X_pca" not in cdata.obsm

    assert ensure_embedding(cdata, "pca", accelerator="cpu") == "X_pca"  # the module: embedded, and kept
    first = cdata.obsm["X_pca"]
    assert first.shape == (40, 3)
    assert ensure_embedding(cdata, "pca", accelerator="cpu") == "X_pca"
    assert cdata.obsm["X_pca"] is first  # not embedded again
    assert ensure_embedding(cdata, "X_pca") == "X_pca"  # also by the obsm key

    with pytest.raises(KeyError, match="neither a key of obsm"):
        ensure_embedding(cdata, "nothing")


def test_a_rep_does_not_replace_an_obsm_entry_that_embed_did_not_make(cdata, trained_pca):
    cdata.obsm["X_pca"] = np.zeros((40, 2))
    with pytest.raises(ValueError, match="not replaced"):
        ensure_embedding(cdata, "pca", accelerator="cpu")
    assert ensure_embedding(cdata, "X_pca") == "X_pca"


def test_a_rep_is_embedded_again_for_a_view_with_a_new_module(cdata, trained_pca):
    embed(cdata, "pca", accelerator="cpu")
    view = cdata[np.arange(5, 35)]
    view.trained_modules["pca"] = cdata.trained_modules["pca"]
    assert ensure_embedding(view, "pca", accelerator="cpu") == "X_pca"
    inherited = view.obsm["X_pca"]  # the module of the parent: current

    highly_variable_genes(view, n_top_genes=10, flavor="seurat", accelerator="cpu")
    pca(view, n_components=3, zscore=True, accelerator="cpu")
    assert ensure_embedding(view, "pca", accelerator="cpu") == "X_pca"
    assert view.obsm["X_pca"] is not inherited  # made again, by the new module
    assert view._cached_embedding_state("pca") == "current"


# ---- the geometric sketch stores the embedding of every cell ----


def _sketch(cdata, **kwargs):
    return geometric_sketch(cdata, target_n_cells=10, embedding_module_key="pca", accelerator="cpu", **kwargs)


def test_the_sketch_stores_the_embedding_of_every_cell_in_the_second_pass(cdata, trained_pca):
    expected = cdata.to_anndata().obsm["X_pca"]  # the embedding of all the cells, computed in memory
    result = _sketch(cdata)

    stored = cdata.obsm["X_pca"]
    assert stored.shape == (40, 3) and stored.dtype == np.float32
    np.testing.assert_allclose(stored, expected, rtol=1e-4, atol=1e-4)
    assert cdata._cached_embedding_state("pca") == "current"  # so it is the embedding that rep="pca" finds
    # and the sketch's own embedding of its cells (from the first pass) agrees with it
    adata = result["adata"]
    np.testing.assert_allclose(
        adata.obsm["X_pca"], stored[adata.obs["cellarium_cell_id"].to_numpy()], rtol=1e-3, atol=1e-3
    )


def test_the_stored_embedding_is_that_of_embed(cdata, trained_pca):
    _sketch(cdata)
    from_the_sketch = np.array(cdata.obsm["X_pca"])
    embed(cdata, "pca", accelerator="cpu")
    np.testing.assert_allclose(from_the_sketch, cdata.obsm["X_pca"], rtol=1e-4, atol=1e-4)


def test_the_embedding_can_be_left_out_of_the_sketch(cdata, trained_pca):
    _sketch(cdata, store_embedding=False)
    assert len(cdata.obsm) == 0


def test_a_sketch_with_a_random_projection_stores_nothing(cdata):
    with pytest.warns(UserWarning, match="random projection"):
        cdata.to_anndata("geometric_sketch", 10, geometric_sketch_kwargs={"accelerator": "cpu"})
    assert len(cdata.obsm) == 0
    geometric_sketch(cdata, target_n_cells=10, accelerator="cpu")
    assert len(cdata.obsm) == 0


def test_a_sketch_does_not_embed_again_when_the_embedding_is_there(cdata, trained_pca):
    embed(cdata, "pca", accelerator="cpu")
    first = cdata.obsm["X_pca"]
    _sketch(cdata)
    assert cdata.obsm["X_pca"] is first
    _sketch(cdata, store_embedding=True)
    assert cdata.obsm["X_pca"] is first


def test_a_sketch_does_not_replace_an_obsm_entry_that_embed_did_not_make(cdata, trained_pca):
    cdata.obsm["X_pca"] = np.zeros((40, 2))
    with pytest.warns(UserWarning, match="not replaced"):
        result = _sketch(cdata)
    assert cdata.obsm["X_pca"].shape == (40, 2)
    assert result["adata"].n_obs == 10  # the sketch itself is made


def test_the_embedding_is_the_whole_embedding_of_the_module_when_the_sketch_uses_some_of_its_pcs(cdata, trained_pca):
    result = _sketch(cdata, n_pcs=2)
    assert cdata.obsm["X_pca"].shape == (40, 3)
    assert result["adata"].n_obs == 10


def test_the_stored_embedding_goes_to_a_file_when_it_does_not_fit(cdata, trained_pca):
    cdata.max_gb = 1e-9
    _sketch(cdata)
    assert isinstance(cdata.obsm["X_pca"], np.memmap)
    assert cdata.obsm["X_pca"].shape == (40, 3)


def test_the_sketch_of_a_view_stores_the_embedding_of_the_cells_of_the_view(cdata, trained_pca):
    positions = np.arange(5, 35)
    view = cdata[positions]
    view.trained_modules["pca"] = cdata.trained_modules["pca"]
    geometric_sketch(view, target_n_cells=8, embedding_module_key="pca", accelerator="cpu")
    assert view.obsm["X_pca"].shape == (30, 3)
    assert "X_pca" not in cdata.obsm
    embed(cdata, "pca", accelerator="cpu")
    np.testing.assert_allclose(view.obsm["X_pca"], cdata.obsm["X_pca"][positions], rtol=1e-4, atol=1e-4)


def test_the_sketch_provides_the_batch_column_of_scvi_when_it_embeds(cdata):
    scvi(cdata, batch_key="cell_type", n_latent=3, max_epochs=1, accelerator="cpu")
    geometric_sketch(cdata, target_n_cells=10, embedding_module_key="scvi", accelerator="cpu")
    assert cdata.obsm["X_scvi"].shape == (40, 3)
    assert "batch_index_n" not in cdata.datamodule.batch_keys  # the datamodule is put back


def test_to_anndata_of_a_sketch_puts_the_stored_embedding_in_the_anndata(cdata, trained_pca):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        adata = cdata.to_anndata("geometric_sketch", 10, geometric_sketch_kwargs={"accelerator": "cpu"})
    assert [str(w.message) for w in record if "obsm['X_pca']" in str(w.message)] == []  # nothing to warn about
    assert cdata._cached_embedding_state("pca") == "current"
    positions = adata.obs["cellarium_cell_id"].to_numpy()
    np.testing.assert_array_equal(adata.obsm["X_pca"], np.asarray(cdata.obsm["X_pca"])[positions])


def test_to_anndata_of_a_sketch_without_the_stored_embedding_computes_it_for_the_sketch_cells(cdata, trained_pca):
    adata = cdata.to_anndata(
        "geometric_sketch", 10, geometric_sketch_kwargs={"accelerator": "cpu", "store_embedding": False}
    )
    assert "X_pca" not in cdata.obsm
    assert adata.obsm["X_pca"].shape == (10, 3)
