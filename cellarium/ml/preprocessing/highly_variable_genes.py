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
    df["means"] = mean_log1p.numpy()
    df["dispersions"] = dispersion.numpy()
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


def get_highly_variable_genes(
    gene_names: list,
    mean: torch.Tensor,
    var: torch.Tensor,
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
        gene_names:
            Ensembl gene ids.
        mean:
            Overall gene expression means in count space (shape ``n_genes``).
        var:
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
        DataFrame indexed by ``gene_names`` with columns ``highly_variable``,
        ``means``, ``dispersions``, ``dispersions_norm``, ``mean_bin``
        (single-batch), ``highly_variable_nbatches`` and
        ``highly_variable_intersection`` (batch mode).
    """
    n_genes = len(gene_names)
    if not (n_genes == len(mean) == len(var)):
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
            gene_names=gene_names,
            mean=mean,
            var=var,
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
    df = _hvg_seurat_single_batch(mean, var, n_bins)
    df.index = gene_names

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
    gene_names: list,
    mean: torch.Tensor,
    var: torch.Tensor,
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
    n_genes = len(gene_names)

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

    df_overall = _hvg_seurat_single_batch(mean, var, n_bins)
    df_overall.index = gene_names

    df_out = pd.DataFrame(index=gene_names)
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
