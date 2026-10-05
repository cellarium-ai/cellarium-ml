# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import anndata as ad
import pandas as pd
import pytest
import torch

from cellarium.ml import CellariumModule
from cellarium.ml.api import CellariumData
from cellarium.ml.api.preprocessing import highly_variable_genes
from cellarium.ml.api.tools import geometric_sketch, pca, scvi
from cellarium.ml.models import IncrementalPCA, SingleCellVariationalInference


@pytest.fixture
def onepass_module(cdata):
    _, module = highly_variable_genes(cdata, n_top_genes=10, flavor="seurat", accelerator="cpu")
    return module


# --- pca() -----------------------------------------------------------------------------------


def test_pca_raises_without_hvg(cdata):
    with pytest.raises(ValueError):
        pca(cdata, n_components=3, zscore=False, accelerator="cpu")


def test_pca_raises_zscore_without_onepass_module(cdata, onepass_module):
    assert cdata.hvg is not None  # onepass_module fixture sets it as a side effect
    with pytest.raises(ValueError):
        pca(cdata, n_components=3, zscore=True, onepass_module=None, accelerator="cpu")


def test_pca_returns_module_with_expected_shape(cdata, onepass_module):
    module = pca(cdata, n_components=3, zscore=True, onepass_module=onepass_module, accelerator="cpu")

    assert isinstance(module, CellariumModule)
    assert isinstance(module.model, IncrementalPCA)
    assert module.model.n_components == 3
    expected_genes = sorted(cdata.hvg.index[cdata.hvg].tolist())
    assert sorted(module.model.var_names_g.tolist()) == expected_genes


# --- geometric_sketch() -----------------------------------------------------------------------


def test_geometric_sketch_raises_without_obs_names_key(cdata):
    del cdata.datamodule.batch_keys["obs_names_n"]
    with pytest.raises(ValueError):
        geometric_sketch(cdata, target_n_cells=10)


def test_geometric_sketch_raises_n_pcs_without_embedding_module(cdata):
    with pytest.raises(ValueError):
        geometric_sketch(cdata, target_n_cells=10, n_pcs=2)


def test_geometric_sketch_raises_n_pcs_with_non_pca_embedding_module(cdata, onepass_module):
    with pytest.raises(ValueError):
        geometric_sketch(cdata, target_n_cells=10, n_pcs=2, embedding_module=onepass_module)


def test_geometric_sketch_default_embedding_returns_expected_keys(cdata):
    n_obs = len(cdata.datamodule.dadc)
    result = geometric_sketch(cdata, target_n_cells=10, return_new_adata=True)

    assert set(result.keys()) == {"obs_names_in_sketch", "adata", "module"}

    sketch_mask = result["obs_names_in_sketch"]
    assert isinstance(sketch_mask, pd.Series)
    assert sketch_mask.dtype == bool
    assert len(sketch_mask) == n_obs
    assert 0 < sketch_mask.sum() <= n_obs

    adata = result["adata"]
    assert isinstance(adata, ad.AnnData)
    assert "X_embedding" in adata.obsm
    assert adata.n_obs == int(sketch_mask.sum())
    assert isinstance(result["module"], CellariumModule)

    # side effect: the sketch mask is also recorded in cdata.obs_computed
    pd.testing.assert_series_equal(cdata.obs_computed["in_sketch"], sketch_mask)


def test_geometric_sketch_with_sparse_coo_batches(make_h5ad_files, monkeypatch):
    # the mps accelerator gets sparse COO instead of CSR batches; force that layout on any platform
    monkeypatch.setattr(torch.mps, "is_available", lambda: True)
    cdata = CellariumData(h5ad_paths=make_h5ad_files(n_files=2, cells_per_file=20, n_genes=30))
    assert next(iter(cdata.datamodule.train_dataloader()))["x_ng"].layout == torch.sparse_coo

    result = geometric_sketch(cdata, target_n_cells=10, return_new_adata=True)

    adata = result["adata"]
    assert isinstance(adata, ad.AnnData)
    assert adata.n_obs == int(result["obs_names_in_sketch"].sum())


def test_geometric_sketch_return_new_adata_false_gives_no_adata(cdata):
    result = geometric_sketch(cdata, target_n_cells=10, return_new_adata=False)
    assert result["adata"] is None
    assert isinstance(result["module"], CellariumModule)


# --- scvi() ------------------------------------------------------------------------------------


def test_scvi_raises_without_batch_key(cdata):
    with pytest.raises(ValueError):
        scvi(cdata, batch_key=None, max_epochs=1, accelerator="cpu")  # type: ignore[arg-type]


def test_scvi_returns_trained_module_and_restores_datamodule_state(cdata):
    datamodule = cdata.datamodule
    original_batch_keys = set(datamodule.batch_keys.keys())
    original_n_train, original_n_val = datamodule.n_train, datamodule.n_val
    original_obs_columns_to_validate = getattr(datamodule.dadc, "obs_columns_to_validate", None)  # h5ad only

    module = scvi(
        cdata,
        batch_key="cell_type",
        n_latent=3,
        max_epochs=2,
        val_size=0.2,
        early_stopping_patience=1,
        accelerator="cpu",
    )

    assert isinstance(module, CellariumModule)
    assert isinstance(module.model, SingleCellVariationalInference)
    assert module.model.n_latent == 3
    assert module.model.n_batch == 3  # "cell_type" has 3 categories in the synthetic fixture

    # side-effect-free: datamodule state restored to what it was before scvi() ran
    assert set(datamodule.batch_keys.keys()) == original_batch_keys
    assert datamodule.n_train == original_n_train
    assert datamodule.n_val == original_n_val
    assert getattr(datamodule.dadc, "obs_columns_to_validate", None) == original_obs_columns_to_validate
