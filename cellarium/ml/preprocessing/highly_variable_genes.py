# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import warnings

import numpy as np
import pandas as pd
import torch


def _hvg_seurat_single_batch(
    mean_g: torch.Tensor,
    var_g: torch.Tensor,
    n_bins: int,
) -> pd.DataFrame:
    """
    Compute binned dispersion statistics for one batch using the Seurat flavor.

    Returns a DataFrame with columns ``means``, ``dispersions``, ``dispersions_norm``,
    ``mean_bin``. Index is a default integer range.
    """
    mean_g = mean_g.clone().float()
    var_g = var_g.clone().float()

    mean_g[mean_g == 0] = 1e-12
    dispersion = var_g / mean_g
    dispersion[dispersion == 0] = np.nan
    dispersion = torch.log(dispersion)
    mean_log1p = torch.log1p(mean_g)

    df = pd.DataFrame()
    df["means"] = mean_log1p.detach().cpu().numpy()
    df["dispersions"] = dispersion.detach().cpu().numpy()
    df["mean_bin"] = pd.cut(df["means"], bins=n_bins)

    disp_grouped = df.groupby("mean_bin", observed=False)["dispersions"]
    disp_mean_bin = disp_grouped.mean()
    disp_std_bin = disp_grouped.std(ddof=1)

    one_gene_per_bin = disp_std_bin.isnull()
    gen_indices = np.where(one_gene_per_bin[df["mean_bin"].values])[0].tolist()
    if len(gen_indices) > 0:
        logging.debug(
            f"Gene indices {gen_indices} fell into a single bin: their "
            "normalized dispersion was set to 1.\n    "
            "Decreasing `n_bins` will likely avoid this effect."
        )
    disp_std_bin[one_gene_per_bin.values] = disp_mean_bin[one_gene_per_bin.values].values
    disp_mean_bin[one_gene_per_bin.values] = 0

    df["dispersions_norm"] = (df["dispersions"].values - disp_mean_bin[df["mean_bin"].values].values) / disp_std_bin[
        df["mean_bin"].values
    ].values

    return df


def seurat_compute_highly_variable_genes(
    var_names_g: list | np.ndarray,
    mean_g: torch.Tensor,
    var_g: torch.Tensor,
    n_top_genes: int | None = None,
    min_disp: float | None = 0.5,
    max_disp: float | None = np.inf,
    min_mean: float | None = 0.0125,
    max_mean: float | None = 3,
    n_bins: int = 20,
    batch_mean_bg: torch.Tensor | None = None,
    batch_var_bg: torch.Tensor | None = None,
    batch_ids: list[str] | None = None,
) -> pd.DataFrame:
    r"""
    Annotate highly variable genes using the ``seurat`` flavor.

    Replicates ``scanpy.pp.highly_variable_genes`` with ``flavor='seurat'``.
    Optionally accepts per-batch statistics for batch-aware selection.

    **References:**

    1. `Highly Variable Genes from Scanpy
       <https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.highly_variable_genes.html>`_.

    Args:
        var_names_g:
            Ensembl gene ids.
        mean_g:
            Overall gene expression means in count space (shape ``n_genes``).
        var_g:
            Overall gene expression variances in count space (shape ``n_genes``).
        n_top_genes:
            Number of highly-variable genes to keep.
        min_disp:
            Ignored when ``n_top_genes`` is set.
        max_disp:
            Ignored when ``n_top_genes`` is set.
        min_mean:
            Ignored when ``n_top_genes`` is set.
        max_mean:
            Ignored when ``n_top_genes`` is set.
        n_bins:
            Number of bins for mean-expression binning.
        batch_mean_bg:
            Per-batch means in count space of shape ``(n_batch, n_genes)``.
        batch_var_bg:
            Per-batch variances in count space of shape ``(n_batch, n_genes)``.
        batch_ids:
            Batch labels of length ``n_batch``.

    Returns:
        DataFrame indexed by ``var_names_g`` with columns ``highly_variable``,
        ``means``, ``dispersions``, ``dispersions_norm``, ``mean_bin``
        (single-batch), ``highly_variable_nbatches`` and
        ``highly_variable_intersection`` (batch mode).
    """
    n_genes = len(var_names_g)
    if not (n_genes == len(mean_g) == len(var_g)):
        raise ValueError("Sizes of `gene_names`, `mean`, and `var` should be the same")

    batch_args = (batch_mean_bg, batch_var_bg, batch_ids)
    if any(a is not None for a in batch_args) and not all(a is not None for a in batch_args):
        raise ValueError("`batch_mean_bg`, `batch_var_bg`, and `batch_ids` must all be provided together.")

    if batch_mean_bg is not None:
        assert batch_var_bg is not None and batch_ids is not None
        n_batch = len(batch_ids)
        if batch_mean_bg.shape != (n_batch, n_genes) or batch_var_bg.shape != (n_batch, n_genes):
            raise ValueError(
                f"`batch_mean_bg` and `batch_var_bg` must have shape (n_batch={n_batch}, n_genes={n_genes})."
            )
        return _get_highly_variable_genes_batched(
            var_names_g=var_names_g,
            mean_g=mean_g,
            var_g=var_g,
            batch_mean_bg=batch_mean_bg,
            batch_var_bg=batch_var_bg,
            batch_ids=batch_ids,
            n_top_genes=n_top_genes,
            min_disp=min_disp,
            max_disp=max_disp,
            min_mean=min_mean,
            max_mean=max_mean,
            n_bins=n_bins,
        )

    # --- Single-batch path ---
    df = _hvg_seurat_single_batch(mean_g, var_g, n_bins)
    df.index = var_names_g

    dispersion_norm = df["dispersions_norm"].values
    mean_log1p = df["means"].values

    if n_top_genes is not None:
        dispersion_norm_nonan = dispersion_norm[~np.isnan(dispersion_norm)]
        dispersion_norm_nonan[::-1].sort()
        if n_top_genes > n_genes:
            logging.info("`n_top_genes` > `adata.n_var`, returning all genes.")
            n_top_genes = n_genes
        if n_top_genes > len(dispersion_norm_nonan):
            warnings.warn(
                "`n_top_genes` > number of normalized dispersions, returning all genes with normalized dispersions.",
                UserWarning,
            )
            n_top_genes = len(dispersion_norm_nonan)
        disp_cut_off = dispersion_norm_nonan[n_top_genes - 1]
        gene_subset = np.nan_to_num(dispersion_norm) >= disp_cut_off
        logging.debug(f"the {n_top_genes} top genes correspond to a normalized dispersion cutoff of {disp_cut_off}")
    else:
        dispersion_norm[np.isnan(dispersion_norm)] = 0  # similar to Seurat
        gene_subset = np.logical_and.reduce(
            (
                mean_log1p > min_mean,
                mean_log1p < max_mean,
                dispersion_norm > min_disp,
                dispersion_norm < max_disp,
            )
        )

    df["highly_variable"] = gene_subset
    return df


def _get_highly_variable_genes_batched(
    var_names_g: list | np.ndarray,
    mean_g: torch.Tensor,
    var_g: torch.Tensor,
    batch_mean_bg: torch.Tensor,
    batch_var_bg: torch.Tensor,
    batch_ids: list[str],
    n_top_genes: int | None,
    min_disp: float | None,
    max_disp: float | None,
    min_mean: float | None,
    max_mean: float | None,
    n_bins: int,
) -> pd.DataFrame:
    """
    Batch-aware HVG selection (seurat flavor). Mirrors
    ``scanpy.pp.highly_variable_genes(..., flavor='seurat', batch_key=...)``.
    """
    n_batch = len(batch_ids)
    n_genes = len(var_names_g)

    per_batch_disp_norm = np.zeros((n_batch, n_genes), dtype=np.float64)
    per_batch_hvg = np.zeros((n_batch, n_genes), dtype=bool)

    for b in range(n_batch):
        df_b = _hvg_seurat_single_batch(batch_mean_bg[b], batch_var_bg[b], n_bins)
        dn = df_b["dispersions_norm"].values.astype(np.float64)
        per_batch_disp_norm[b] = np.nan_to_num(dn, nan=0.0)

        if n_top_genes is not None:
            dn_nonan = dn[~np.isnan(dn)]
            if len(dn_nonan) == 0:
                continue
            n_select = min(n_top_genes, len(dn_nonan))
            cut = np.sort(dn_nonan)[::-1][n_select - 1]
            per_batch_hvg[b] = np.nan_to_num(dn) >= cut
        else:
            means_b = df_b["means"].values
            per_batch_hvg[b] = np.logical_and.reduce(
                (
                    means_b > min_mean,
                    means_b < max_mean,
                    dn > min_disp,
                    dn < max_disp,
                )
            )

    highly_variable_nbatches = per_batch_hvg.sum(axis=0).astype(int)
    dispersions_norm_mean = per_batch_disp_norm.mean(axis=0)

    df_overall = _hvg_seurat_single_batch(mean_g, var_g, n_bins)
    df_overall.index = var_names_g

    df_out = pd.DataFrame(index=var_names_g)
    df_out["means"] = df_overall["means"].values
    df_out["dispersions"] = df_overall["dispersions"].values
    df_out["dispersions_norm"] = dispersions_norm_mean
    df_out["highly_variable_nbatches"] = highly_variable_nbatches
    df_out["highly_variable_intersection"] = highly_variable_nbatches == n_batch

    if n_top_genes is not None:
        orig_index = df_out.index.copy()
        df_out = df_out.sort_values(
            ["highly_variable_nbatches", "dispersions_norm"],
            ascending=False,
            na_position="last",
        )
        df_out["highly_variable"] = np.arange(len(df_out)) < n_top_genes
        df_out = df_out.loc[orig_index]
    else:
        df_out["dispersions_norm"] = df_out["dispersions_norm"].fillna(0)
        df_out["highly_variable"] = np.logical_and.reduce(
            (
                df_out["means"].values > min_mean,
                df_out["means"].values < max_mean,
                df_out["dispersions_norm"].values > min_disp,
                df_out["dispersions_norm"].values < max_disp,
            )
        )

    return df_out


def kotliar_compute_highly_variable_genes(
    var_names_g: list | np.ndarray,
    mean_g: np.ndarray | torch.Tensor,
    var_g: np.ndarray | torch.Tensor,
    n_top_genes: int | None = 2000,
    expected_fano_threshold: float | None = None,
    minimal_mean: float = 0.5,
    plot: bool = False,
):
    """
    Helper function to run the highly variable gene selection procedure from Kotliar et al. 2019,
    implemented in the function ``get_highvar_genes_sparse`` in the dylkot/cNMF repository.
    Modified to work in cellarium based on a run of a onepass_mean_var_std model on the data.

    NOTE: taken from
    https://github.com/dylkot/cNMF/blob/5dbc5baaa0b9079b55bce554d801caa235a50457/src/cnmf/cnmf.py#L136-L188

    Args:
        mean_g: The mean expression levels of genes
        var_g: The variance of expression levels of genes
        var_names_g: The names of the genes
        n_top_genes: The number of highly variable genes to select. If None, uses a threshold-based approach
        expected_fano_threshold: If n_top_genes is None, this threshold is used to select highly variable genes
            based on their Fano factor relative to the expected Fano factor. If None, a default threshold is
            computed based on the standard deviation of the Fano factors of genes that pass a winsorized box filter.
        minimal_mean: The minimum mean expression level for a gene to be considered highly variable.
            This is used only in the threshold-based approach (i.e. when n_top_genes is None)
        plot: Whether to plot the mean-variance relationship and the Fano factor distribution. Useful for debugging.

    Returns:
        A DataFrame with columns
        - mean: The mean expression level of each gene
        - var: The variance of each gene
        - fano: The Fano factor of each gene (variance / mean)
        - fano_fit: The expected Fano factor of each gene based on the fitted line
        - fano_ratio: The ratio of the observed Fano factor to the expected Fano factor
        - highly_variable: A boolean indicating whether the gene is selected as highly variable
    """

    df = pd.DataFrame({"mean_g": mean_g, "var_g": var_g, "fano_g": var_g / mean_g}, index=var_names_g)

    # Find parameters for expected fano line
    top_genes = df["mean_g"].sort_values(ascending=False)[:20].index
    A = (np.sqrt(df["var_g"]) / df["mean_g"])[top_genes].min()

    w_mean_low, w_mean_high = df["mean_g"].quantile([0.10, 0.90])
    w_fano_low, w_fano_high = df["fano_g"].quantile([0.10, 0.90])
    winsor_box_logic = (
        (df["fano_g"] > w_fano_low)
        & (df["fano_g"] < w_fano_high)
        & (df["mean_g"] > w_mean_low)
        & (df["mean_g"] < w_mean_high)
    )
    fano_median = df["fano_g"][winsor_box_logic].median()
    B = np.sqrt(fano_median)

    df["fano_fit_g"] = (A**2) * df["mean_g"] + (B**2)
    df["fano_ratio_g"] = df["fano_g"] / df["fano_fit_g"]

    # Identify high var genes
    if n_top_genes is not None:
        hvg_var_names = df["fano_ratio_g"].sort_values(ascending=False).index[:n_top_genes]
        hvg_logic_g = df.index.isin(hvg_var_names)
        T = None
    else:
        if not expected_fano_threshold:
            T = 1.0 + df["fano_g"][winsor_box_logic].std()
        else:
            T = expected_fano_threshold
        hvg_logic_g = (df["fano_ratio_g"] > T) & (df["mean_g"] > minimal_mean)

    df["highly_variable_g"] = hvg_logic_g

    if plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 3.5))
        plt.subplot(1, 3, 1)
        plt.scatter(df["mean_g"], df["var_g"], s=2, alpha=1, color="lightgray", label="All genes")
        plt.scatter(
            df["mean_g"][hvg_logic_g],
            df["var_g"][hvg_logic_g],
            s=4,
            alpha=0.2,
            color="r",
            label="Highly variable genes",
        )
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Mean expression")
        plt.ylabel("Variance of expression")
        plt.title("Gene mean vs. variance")
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.scatter(df["mean_g"], df["fano_g"], s=2, alpha=1, color="lightgray", label="All genes")
        plt.scatter(
            df["mean_g"][hvg_logic_g],
            df["fano_g"][hvg_logic_g],
            s=4,
            alpha=0.2,
            color="r",
            label="Highly variable genes",
        )
        order = np.argsort(df["mean_g"])
        plt.plot(df["mean_g"][order], df["fano_fit_g"][order], color="k", linestyle="--")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Mean expression")
        plt.ylabel("Fano factor")
        plt.title("Gene mean vs. Fano factor")
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.scatter(df["mean_g"], df["fano_ratio_g"], s=2, alpha=1, color="lightgray", label="All genes")
        plt.scatter(
            df["mean_g"][hvg_logic_g],
            df["fano_ratio_g"][hvg_logic_g],
            s=4,
            alpha=0.2,
            color="r",
            label="Highly variable genes",
        )
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Mean expression")
        plt.ylabel("Fano ratio")
        plt.title("Gene mean vs. Fano ratio")
        plt.legend()
        plt.tight_layout()
        plt.show()

    df = df.rename(columns={c: c.split("_g")[0] for c in df.columns})
    return df
