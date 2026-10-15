# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellarium.ml import CellariumModule
from cellarium.ml.api.preprocessing import highly_variable_genes, pseudobulk
from cellarium.ml.models import HVGSeuratV3


def test_highly_variable_genes_seurat_sets_hvg_and_returns_df(cdata):
    hvg_df, module = highly_variable_genes(cdata, n_top_genes=10, flavor="seurat", accelerator="cpu")

    assert isinstance(hvg_df, pd.DataFrame)
    assert "highly_variable" in hvg_df.columns
    assert list(hvg_df.index) == list(cdata.datamodule.var_names_g)
    assert isinstance(module, CellariumModule)

    # highly_variable_genes() sets cdata.hvg as a side effect, aligned to the returned mask
    pd.testing.assert_series_equal(cdata.hvg, hvg_df["highly_variable"].astype(bool), check_names=False)


@pytest.mark.parametrize("flavor", ["seurat_v3", "kotliar"])
def test_highly_variable_genes_other_flavors_smoke(cdata, flavor):
    hvg_df, module = highly_variable_genes(cdata, n_top_genes=10, flavor=flavor, accelerator="cpu")

    assert "highly_variable" in hvg_df.columns
    assert len(hvg_df) == len(cdata.datamodule.var_names_g)
    assert isinstance(module, CellariumModule)


def test_highly_variable_genes_invalid_flavor_raises(cdata):
    with pytest.raises(ValueError):
        highly_variable_genes(cdata, flavor="bogus", accelerator="cpu")  # type: ignore[arg-type]


@pytest.mark.parametrize("flavor", ["seurat", "kotliar"])
def test_highly_variable_genes_batch_key_unsupported_flavor_raises(cdata, flavor):
    with pytest.raises(ValueError):
        highly_variable_genes(cdata, n_top_genes=10, flavor=flavor, batch_key="cell_type", accelerator="cpu")


def test_highly_variable_genes_seurat_v3_with_batch_key_trains_and_restores_datamodule_state(cdata):
    datamodule = cdata.datamodule
    original_batch_keys = set(datamodule.batch_keys.keys())
    original_obs_columns_to_validate = datamodule.dadc.obs_columns_to_validate

    hvg_df, module = highly_variable_genes(
        cdata, n_top_genes=10, flavor="seurat_v3", batch_key="cell_type", accelerator="cpu"
    )

    assert isinstance(module.model, HVGSeuratV3)
    assert module.model.use_batch_key is True
    assert module.model.n_batch == 3  # "cell_type" has 3 categories in the synthetic fixture
    assert "highly_variable_nbatches" in hvg_df.columns

    # side-effect-free: datamodule state restored to what it was before highly_variable_genes() ran
    assert set(datamodule.batch_keys.keys()) == original_batch_keys
    assert datamodule.dadc.obs_columns_to_validate == original_obs_columns_to_validate


# --- pseudobulk() ------------------------------------------------------------------------------


def test_pseudobulk_raises_without_batch_key(cdata):
    with pytest.raises(ValueError):
        pseudobulk(cdata, batch_key=None, accelerator="cpu")  # type: ignore[arg-type]


def test_pseudobulk_returns_expected_anndata_and_restores_datamodule_state(cdata):
    datamodule = cdata.datamodule
    original_batch_keys = set(datamodule.batch_keys.keys())
    original_obs_columns_to_validate = datamodule.dadc.obs_columns_to_validate

    result = pseudobulk(cdata, batch_key="cell_type", accelerator="cpu")

    assert isinstance(result, ad.AnnData)
    assert result.n_vars == len(cdata.datamodule.var_names_g)
    assert result.n_obs == 3  # "cell_type" has 3 categories in the synthetic fixture
    assert list(result.var_names) == list(cdata.datamodule.var_names_g)
    assert set(result.obs_names) == {"T cell", "B cell", "NK cell"}
    assert "n_cells" in result.obs.columns
    assert result.obs["n_cells"].sum() == len(cdata.datamodule.dadc)

    assert "pseudobulk_mean" in result.layers
    assert "pseudobulk_std" in result.layers
    np.testing.assert_array_equal(result.X, result.layers["pseudobulk_mean"])
    assert result.layers["pseudobulk_mean"].shape == (3, len(cdata.datamodule.var_names_g))
    assert result.layers["pseudobulk_std"].shape == (3, len(cdata.datamodule.var_names_g))

    # side-effect-free: datamodule state restored to what it was before pseudobulk() ran
    assert set(datamodule.batch_keys.keys()) == original_batch_keys
    assert datamodule.dadc.obs_columns_to_validate == original_obs_columns_to_validate


def test_pseudobulk_zscore_smoke(cdata):
    result = pseudobulk(cdata, batch_key="cell_type", normalize_total=True, log1p=True, zscore=True, accelerator="cpu")

    assert result.n_obs == 3
    assert result.n_vars == len(cdata.datamodule.var_names_g)
