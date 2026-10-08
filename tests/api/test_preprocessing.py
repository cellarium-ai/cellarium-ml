# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import torch

from cellarium.ml import CellariumModule
from cellarium.ml.api import CellariumData
from cellarium.ml.api.preprocessing import highly_variable_genes, pseudobulk
from cellarium.ml.models import HVGSeuratV3, OnePassMeanVarStd


def _read_counts(cdata: CellariumData) -> ad.AnnData:
    """The counts of all the cells of `cdata`, as an AnnData of its genes, read directly from its datamodule."""
    dadc = cdata.datamodule.dadc
    adata = dadc[: len(dadc)]
    return ad.AnnData(
        X=adata.X.toarray().astype(np.float64),
        obs=adata.obs[["cell_type"]].astype(str).astype("category"),
        var=pd.DataFrame(index=cdata.datamodule.var_names_g),
    )


def test_highly_variable_genes_seurat_sets_hvg_and_returns_df(cdata):
    hvg_df = highly_variable_genes(cdata, n_top_genes=10, flavor="seurat", accelerator="cpu")

    assert isinstance(hvg_df, pd.DataFrame)
    assert "highly_variable" in hvg_df.columns
    assert list(hvg_df.index) == list(cdata.datamodule.var_names_g)
    assert isinstance(cdata.trained_modules["onepass"].module, CellariumModule)

    # highly_variable_genes() sets cdata.hvg as a side effect, aligned to the returned mask
    pd.testing.assert_series_equal(cdata.hvg, hvg_df["highly_variable"].astype(bool), check_names=False)


@pytest.mark.parametrize("flavor", ["seurat_v3", "kotliar"])
def test_highly_variable_genes_other_flavors_smoke(cdata, flavor):
    hvg_df = highly_variable_genes(cdata, n_top_genes=10, flavor=flavor, accelerator="cpu")

    assert "highly_variable" in hvg_df.columns
    assert len(hvg_df) == len(cdata.datamodule.var_names_g)
    (trained,) = cdata.trained_modules.values()
    assert isinstance(trained.module, CellariumModule)


@pytest.mark.parametrize(
    "flavor, expected_key", [("seurat", "onepass"), ("kotliar", "onepass"), ("seurat_v3", "hvg_seurat_v3")]
)
def test_highly_variable_genes_registers_trained_module(cdata, flavor, expected_key):
    highly_variable_genes(cdata, n_top_genes=10, flavor=flavor, accelerator="cpu")

    assert set(cdata.trained_modules) == {expected_key}
    trained = cdata.trained_modules[expected_key]
    assert isinstance(trained.module, CellariumModule)
    assert trained.complete
    assert trained.history.empty
    assert expected_key in repr(cdata)


def test_highly_variable_genes_key_added(cdata):
    highly_variable_genes(cdata, n_top_genes=10, flavor="seurat", key_added="my_onepass", accelerator="cpu")
    assert set(cdata.trained_modules) == {"my_onepass"}


def test_highly_variable_genes_invalid_flavor_raises(cdata):
    with pytest.raises(ValueError):
        highly_variable_genes(cdata, flavor="bogus", accelerator="cpu")  # type: ignore[arg-type]


def test_highly_variable_genes_batch_key_unsupported_flavor_raises(cdata):
    with pytest.raises(ValueError):
        highly_variable_genes(cdata, n_top_genes=10, flavor="kotliar", batch_key="cell_type", accelerator="cpu")


def test_highly_variable_genes_seurat_with_batch_key_trains_and_restores_datamodule_state(cdata):
    datamodule = cdata.datamodule
    original_batch_keys = set(datamodule.batch_keys.keys())
    original_obs_columns_to_validate = getattr(datamodule.dadc, "obs_columns_to_validate", None)  # h5ad only

    hvg_df = highly_variable_genes(cdata, n_top_genes=10, flavor="seurat", batch_key="cell_type", accelerator="cpu")

    module = cdata.trained_modules["onepass"].module
    assert isinstance(module.model, OnePassMeanVarStd)
    assert module.model.n_batch == 3  # "cell_type" has 3 categories in the synthetic fixture
    assert "highly_variable_nbatches" in hvg_df.columns
    assert hvg_df["highly_variable"].sum() == 10
    assert cdata.trained_modules["onepass"].config["batch_key"] == "cell_type"

    # side-effect-free: datamodule state restored to what it was before highly_variable_genes() ran
    assert set(datamodule.batch_keys.keys()) == original_batch_keys
    assert getattr(datamodule.dadc, "obs_columns_to_validate", None) == original_obs_columns_to_validate


@pytest.mark.parametrize("batch_key", [None, "cell_type"])
def test_highly_variable_genes_seurat_matches_scanpy(cdata, batch_key):
    n_top_genes = 10
    adata = _read_counts(cdata)
    sc.pp.normalize_total(adata, target_sum=10_000)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=n_top_genes, batch_key=batch_key)

    hvg_df = highly_variable_genes(
        cdata, n_top_genes=n_top_genes, flavor="seurat", batch_key=batch_key, accelerator="cpu"
    )

    model = cdata.trained_modules["onepass"].module.model
    # scanpy's "means" are log1p of the means of the normalized counts and, with batches, averaged over the batches
    log_means = torch.log1p(model.mean_g) if batch_key is None else torch.log1p(model.batch_mean_bg).mean(dim=0)
    np.testing.assert_allclose(log_means.numpy(), adata.var["means"], rtol=1e-4)
    np.testing.assert_allclose(hvg_df["dispersions_norm"], adata.var["dispersions_norm"], rtol=1e-3, atol=1e-4)
    if batch_key is None:
        assert set(hvg_df.index[hvg_df["highly_variable"]]) == set(adata.var_names[adata.var["highly_variable"]])


# --- fits are reused ----------------------------------------------------------------------------


def test_highly_variable_genes_seurat_and_kotliar_share_one_onepass_fit(cdata, fits):
    highly_variable_genes(cdata, n_top_genes=10, flavor="seurat", accelerator="cpu")
    trained = cdata.trained_modules["onepass"]
    highly_variable_genes(cdata, n_top_genes=7, flavor="kotliar", accelerator="cpu")
    highly_variable_genes(cdata, n_top_genes=5, flavor="seurat", accelerator="cpu", key_added="again")

    assert len(fits) == 1
    assert cdata.trained_modules["onepass"] is trained
    assert cdata.trained_modules["again"] is trained
    assert trained.config == {"target_count": 10_000, "eps": 1e-6, "log1p": False, "batch_key": None}


def test_highly_variable_genes_onepass_with_batches_serves_a_call_without_but_not_the_reverse(cdata, fits):
    highly_variable_genes(cdata, n_top_genes=10, flavor="seurat", accelerator="cpu")
    highly_variable_genes(cdata, n_top_genes=10, flavor="seurat", batch_key="cell_type", accelerator="cpu")
    assert [config["batch_key"] for config in fits] == [None, "cell_type"]

    unbatched_df = highly_variable_genes(cdata, n_top_genes=10, flavor="kotliar", accelerator="cpu")
    assert len(fits) == 2
    assert "highly_variable_nbatches" not in unbatched_df.columns


def test_highly_variable_genes_unbatched_after_batched_matches_a_fresh_fit(cdata, fits):
    highly_variable_genes(cdata, n_top_genes=10, flavor="seurat", batch_key="cell_type", accelerator="cpu")
    reused_df = highly_variable_genes(cdata, n_top_genes=10, flavor="seurat", accelerator="cpu")
    assert len(fits) == 1

    cdata._fit_cache.clear()
    fresh_df = highly_variable_genes(cdata, n_top_genes=10, flavor="seurat", accelerator="cpu")
    assert len(fits) == 2

    pd.testing.assert_frame_equal(reused_df, fresh_df, rtol=1e-4)


def test_highly_variable_genes_seurat_v3_reused_for_other_n_top_genes_but_exact_on_batch_key(cdata, fits):
    df_10 = highly_variable_genes(cdata, n_top_genes=10, flavor="seurat_v3", accelerator="cpu")
    df_5 = highly_variable_genes(cdata, n_top_genes=5, flavor="seurat_v3", accelerator="cpu")
    assert len(fits) == 1
    assert df_10["highly_variable"].sum() == 10
    assert df_5["highly_variable"].sum() == 5

    highly_variable_genes(cdata, n_top_genes=10, flavor="seurat_v3", batch_key="cell_type", accelerator="cpu")
    highly_variable_genes(cdata, n_top_genes=10, flavor="seurat_v3", accelerator="cpu")
    assert [config["batch_key"] for config in fits] == [None, "cell_type"]


@pytest.mark.parametrize("flavor", ["seurat", "seurat_v3"])
def test_highly_variable_genes_writes_no_csv(cdata, tmp_path, monkeypatch, flavor):
    monkeypatch.chdir(tmp_path)
    highly_variable_genes(cdata, n_top_genes=10, flavor=flavor, accelerator="cpu")
    assert list(tmp_path.glob("*.csv")) == []


def test_highly_variable_genes_seurat_v3_with_batch_key_trains_and_restores_datamodule_state(cdata):
    datamodule = cdata.datamodule
    original_batch_keys = set(datamodule.batch_keys.keys())
    original_obs_columns_to_validate = getattr(datamodule.dadc, "obs_columns_to_validate", None)  # h5ad only

    hvg_df = highly_variable_genes(cdata, n_top_genes=10, flavor="seurat_v3", batch_key="cell_type", accelerator="cpu")

    module = cdata.trained_modules["hvg_seurat_v3"].module
    assert isinstance(module.model, HVGSeuratV3)
    assert module.model.use_batch_key is True
    assert module.model.n_batch == 3  # "cell_type" has 3 categories in the synthetic fixture
    assert "highly_variable_nbatches" in hvg_df.columns

    # side-effect-free: datamodule state restored to what it was before highly_variable_genes() ran
    assert set(datamodule.batch_keys.keys()) == original_batch_keys
    assert getattr(datamodule.dadc, "obs_columns_to_validate", None) == original_obs_columns_to_validate


# --- pseudobulk() ------------------------------------------------------------------------------


def test_pseudobulk_raises_without_batch_key(cdata):
    with pytest.raises(ValueError):
        pseudobulk(cdata, batch_key=None, accelerator="cpu")  # type: ignore[arg-type]


def test_pseudobulk_returns_expected_anndata_and_restores_datamodule_state(cdata):
    datamodule = cdata.datamodule
    original_batch_keys = set(datamodule.batch_keys.keys())
    original_obs_columns_to_validate = getattr(datamodule.dadc, "obs_columns_to_validate", None)  # h5ad only

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
    assert getattr(datamodule.dadc, "obs_columns_to_validate", None) == original_obs_columns_to_validate


def test_pseudobulk_zscore_smoke(cdata):
    result = pseudobulk(cdata, batch_key="cell_type", normalize_total=True, log1p=True, zscore=True, accelerator="cpu")

    assert result.n_obs == 3
    assert result.n_vars == len(cdata.datamodule.var_names_g)
