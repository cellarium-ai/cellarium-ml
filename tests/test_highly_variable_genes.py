# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import scipy.sparse
import torch

from cellarium.ml.preprocessing.highly_variable_genes import (
    kotliar_compute_highly_variable_genes,
    seurat_compute_highly_variable_genes,
)

# ---------------------------------------------------------------------------
# Original smoke tests (preserved exactly)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_genes_to_check", [2, 3, 4])
def test_highly_variable_genes_top_n(num_genes_to_check: int):
    gene_names = ["gene_0", "gene_1", "gene_2", "gene_3", "gene_4", "gene_5"]
    mean = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32)
    var = torch.tensor([1.6086e00, 2.2582e02, 9.8922e-02, 1.4379e01, 4.0901e-02, 9.9200e-01], dtype=torch.float32)
    result = seurat_compute_highly_variable_genes(
        var_names_g=gene_names, mean_g=mean, var_g=var, n_top_genes=num_genes_to_check, n_bins=1
    )
    assert result[result.highly_variable].shape[0] == num_genes_to_check


def test_highly_variable_genes_cutoffs():
    gene_names = ["gene_0", "gene_1", "gene_2", "gene_3", "gene_4", "gene_5"]
    mean = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32)
    var = torch.tensor([1.6086e00, 2.2582e02, 9.8922e-02, 1.4379e01, 4.0901e-02, 9.9200e-01], dtype=torch.float32)
    result = seurat_compute_highly_variable_genes(
        gene_names, mean, var, min_disp=0.2, max_disp=10, min_mean=0, max_mean=5, n_bins=1
    )
    assert result[result.highly_variable].shape[0] != 0


def test_highly_variable_genes_wrong_sizes():
    gene_names = ["gene_0", "gene_1", "gene_2", "gene_3"]
    mean = torch.tensor([1, 2, 3])
    var = torch.tensor([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError):
        seurat_compute_highly_variable_genes(gene_names, mean, var)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_adata():
    import scanpy as sc

    adata, batch_column = sc.datasets.ebi_expression_atlas("E-MTAB-10137"), "Sample Characteristic[individual]"
    adata.obs[batch_column] = adata.obs[batch_column].astype("category")
    return adata, batch_column


def _normalize_log1p_adata(adata):
    """Normalize to median total count then log1p-transform: the expected
    input for the seurat (non-v3) HVG flavor."""
    import scanpy as sc

    adata = adata.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return adata


def _to_dense(x):
    return x.toarray() if scipy.sparse.issparse(x) else np.asarray(x)


def _mean_var_ddof1(x: np.ndarray):
    """Sample mean + variance (ddof=1), matching scanpy's fast_array_utils.stats.mean_var."""
    n = x.shape[0]
    mean = x.mean(axis=0)
    var = x.var(axis=0) * n / (n - 1)
    return mean.astype(np.float64), var.astype(np.float64)


# ---------------------------------------------------------------------------
# Phase 1: Numerical comparison against Scanpy (single-batch)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_top_genes", [200, 500])
def test_hvg_matches_scanpy_n_top_genes(n_top_genes: int):
    """Verify seurat-flavor HVG output matches scanpy.pp.highly_variable_genes."""
    import scanpy as sc

    adata, _ = _load_adata()
    adata_norm = _normalize_log1p_adata(adata)
    # Scanpy's seurat HVG internally reverses log1p (expm1) before computing
    # mean/var, so we must pass statistics from normalized count space.
    x_counts = np.expm1(_to_dense(adata_norm.X))
    mean_np, var_np = _mean_var_ddof1(x_counts)
    mean_t = torch.tensor(mean_np, dtype=torch.float32)
    var_t = torch.tensor(var_np, dtype=torch.float32)

    our_df = seurat_compute_highly_variable_genes(list(adata_norm.var_names), mean_t, var_t, n_top_genes=n_top_genes)
    sc_df = sc.pp.highly_variable_genes(adata_norm, flavor="seurat", n_top_genes=n_top_genes, inplace=False)

    our_hvg = set(our_df[our_df.highly_variable].index)
    sc_hvg = set(sc_df[sc_df.highly_variable].index)
    jaccard = len(our_hvg & sc_hvg) / len(our_hvg | sc_hvg)
    assert jaccard >= 0.99, f"Jaccard {jaccard:.4f} below 0.99 for n_top_genes={n_top_genes}"

    common = list(set(our_df.index) & set(sc_df.index))
    np.testing.assert_allclose(
        our_df.loc[common, "dispersions_norm"].values.astype(float),
        sc_df.loc[common, "dispersions_norm"].values.astype(float),
        atol=1e-2,
        rtol=1e-3,
        err_msg="dispersions_norm values differ more than expected from Scanpy",
    )


def test_hvg_matches_scanpy_cutoffs():
    """Verify cutoff-based selection matches Scanpy with default cutoffs."""
    import scanpy as sc

    adata, _ = _load_adata()
    adata_norm = _normalize_log1p_adata(adata)
    x_counts = np.expm1(_to_dense(adata_norm.X))
    mean_np, var_np = _mean_var_ddof1(x_counts)
    mean_t = torch.tensor(mean_np, dtype=torch.float32)
    var_t = torch.tensor(var_np, dtype=torch.float32)

    our_df = seurat_compute_highly_variable_genes(
        list(adata_norm.var_names), mean_t, var_t, min_disp=0.5, max_disp=np.inf, min_mean=0.0125, max_mean=3.0
    )
    sc_df = sc.pp.highly_variable_genes(
        adata_norm, flavor="seurat", min_disp=0.5, max_disp=np.inf, min_mean=0.0125, max_mean=3.0, inplace=False
    )

    our_hvg = set(our_df[our_df.highly_variable].index)
    sc_hvg = set(sc_df[sc_df.highly_variable].index)
    jaccard = len(our_hvg & sc_hvg) / max(len(our_hvg | sc_hvg), 1)
    assert jaccard >= 0.99, f"Jaccard {jaccard:.4f} (cutoff mode) below 0.99"


# ---------------------------------------------------------------------------
# Phase 2: Batch-aware tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_top_genes", [200, 500])
def test_hvg_batch_matches_scanpy(n_top_genes: int):
    """Verify batch HVG selection matches scanpy's batch_key path."""
    import scanpy as sc

    adata, batch_col = _load_adata()
    adata_norm = _normalize_log1p_adata(adata)
    x_counts = np.expm1(_to_dense(adata_norm.X))
    batches = list(adata_norm.obs[batch_col].cat.categories)
    n_batch = len(batches)
    n_genes = adata_norm.n_vars

    mean_np, var_np = _mean_var_ddof1(x_counts)
    mean_t = torch.tensor(mean_np, dtype=torch.float32)
    var_t = torch.tensor(var_np, dtype=torch.float32)

    batch_mean_bg = torch.zeros(n_batch, n_genes, dtype=torch.float32)
    batch_var_bg = torch.zeros(n_batch, n_genes, dtype=torch.float32)
    for i, b in enumerate(batches):
        mask = (adata_norm.obs[batch_col] == b).values
        bm, bv = _mean_var_ddof1(x_counts[mask])
        batch_mean_bg[i] = torch.tensor(bm, dtype=torch.float32)
        batch_var_bg[i] = torch.tensor(bv, dtype=torch.float32)

    our_df = seurat_compute_highly_variable_genes(
        list(adata_norm.var_names),
        mean_t,
        var_t,
        n_top_genes=n_top_genes,
        batch_mean_bg=batch_mean_bg,
        batch_var_bg=batch_var_bg,
        batch_ids=batches,
    )
    sc_df = sc.pp.highly_variable_genes(
        adata_norm, flavor="seurat", n_top_genes=n_top_genes, batch_key=batch_col, inplace=False
    )

    our_hvg = set(our_df[our_df.highly_variable].index)
    sc_hvg = set(sc_df[sc_df.highly_variable].index)
    jaccard = len(our_hvg & sc_hvg) / len(our_hvg | sc_hvg)
    assert jaccard >= 0.99, f"Batch Jaccard {jaccard:.4f} below 0.99 for n_top_genes={n_top_genes}"

    assert "highly_variable_nbatches" in our_df.columns
    assert our_df["highly_variable_nbatches"].between(0, n_batch).all()
    assert "highly_variable_intersection" in our_df.columns
    assert set(our_df[our_df.highly_variable_intersection].index).issubset(our_hvg)


def test_hvg_batch_partial_args_raises():
    with pytest.raises(ValueError, match="must all be provided together"):
        seurat_compute_highly_variable_genes(["g0", "g1"], torch.ones(2), torch.ones(2), batch_mean_bg=torch.ones(2, 2))


def test_hvg_batch_shape_mismatch_raises():
    with pytest.raises(ValueError):
        seurat_compute_highly_variable_genes(
            ["g0", "g1", "g2"],
            torch.ones(3),
            torch.ones(3),
            batch_mean_bg=torch.ones(2, 4),
            batch_var_bg=torch.ones(2, 4),
            batch_ids=["a", "b"],
        )


# ---------------------------------------------------------------------------
# Kotliar HVG tests
# ---------------------------------------------------------------------------

_KOTLIAR_GENE_NAMES = [f"gene_{i}" for i in range(50)]
# Construct data with clear high-variance outliers: first 5 genes have much
# higher variance than the rest.
_rng = np.random.default_rng(42)
_KOTLIAR_MEAN = np.abs(_rng.normal(loc=5.0, scale=2.0, size=50)) + 0.1
_KOTLIAR_VAR = _KOTLIAR_MEAN * 1.1          # Fano ≈ 1 baseline
_KOTLIAR_VAR[:5] = _KOTLIAR_MEAN[:5] * 50  # very high Fano for first 5 genes


@pytest.mark.parametrize("num_genes", [5, 10, 20])
def test_kotliar_num_genes_exact(num_genes: int):
    """num_genes selected genes are returned when num_genes is specified."""
    df = kotliar_compute_highly_variable_genes(
        mean_g=_KOTLIAR_MEAN,
        var_g=_KOTLIAR_VAR,
        var_names_g=_KOTLIAR_GENE_NAMES,
        num_genes=num_genes,
    )
    assert df["highly_variable"].sum() == num_genes


def test_kotliar_output_schema():
    """Output DataFrame has the required columns and is indexed by var_names_g."""
    df = kotliar_compute_highly_variable_genes(
        mean_g=_KOTLIAR_MEAN,
        var_g=_KOTLIAR_VAR,
        var_names_g=_KOTLIAR_GENE_NAMES,
        num_genes=10,
    )
    expected_cols = {"mean", "var", "fano", "fano_fit", "fano_ratio", "highly_variable"}
    assert expected_cols.issubset(set(df.columns)), f"Missing columns: {expected_cols - set(df.columns)}"
    assert list(df.index) == _KOTLIAR_GENE_NAMES


def test_kotliar_top_genes_have_highest_fano_ratio():
    """Genes selected by num_genes should be exactly those with the highest fano_ratio."""
    num_genes = 10
    df = kotliar_compute_highly_variable_genes(
        mean_g=_KOTLIAR_MEAN,
        var_g=_KOTLIAR_VAR,
        var_names_g=_KOTLIAR_GENE_NAMES,
        num_genes=num_genes,
    )
    top_by_ratio = df["fano_ratio"].nlargest(num_genes).index
    selected = df[df["highly_variable"]].index
    assert set(selected) == set(top_by_ratio)


def test_kotliar_threshold_mode_selects_outliers():
    """With a high threshold, only the obvious high-Fano genes are selected."""
    df = kotliar_compute_highly_variable_genes(
        mean_g=_KOTLIAR_MEAN,
        var_g=_KOTLIAR_VAR,
        var_names_g=_KOTLIAR_GENE_NAMES,
        num_genes=None,
        expected_fano_threshold=20.0,   # only fano_ratio > 20 pass
        minimal_mean=0.0,
    )
    # All selected genes must have fano_ratio > 20
    selected = df[df["highly_variable"]]
    assert (selected["fano_ratio"] > 20.0).all()
    # The 5 genes we engineered to have Fano ≈ 50× the fit should all be selected
    assert set(_KOTLIAR_GENE_NAMES[:5]).issubset(set(selected.index))
