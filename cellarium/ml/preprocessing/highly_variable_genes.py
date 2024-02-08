# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import warnings

import numpy as np
import pandas as pd
import torch


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
) -> pd.DataFrame:
    r"""
    Get Highly variably genes. This is a replication of Highly Variable Genes from Scanpy with a ``Seurat`` flavor.

    **References:**

    1. `Highly Variable Genes from Scanpy
       <https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.highly_variable_genes.html>`_.

    Args:
        gene_names:
            Ensembl gene ids.
        mean:
            Gene expression means.
        var:
            Gene expression vars.
        n_top_genes:
            Number of highly-variable genes to keep.
        min_disp:
            If ``n_top_genes`` unequals None, this and all other cutoffs for the means and the normalized
            dispersions are ignored.
        max_disp:
            If ``n_top_genes`` unequals None, this and all other cutoffs for the means and the normalized
            dispersions are ignored.
        min_mean:
            If ``n_top_genes`` unequals None, this and all other cutoffs for the means and the normalized
            dispersions are ignored.
        max_mean:
            If ``n_top_genes`` unequals None, this and all other cutoffs for the means and the normalized
            dispersions are ignored.
        n_bins:
            Number of bins for binning the mean gene expression. Normalization is done with respect to each bin.
            If just a single gene falls into a bin, the normalized dispersion is artificially set to 1. You’ll be
            informed about this
    """
    # compute the dispersion
    if not (len(gene_names) == len(mean) == len(var)):
        raise ValueError("Sizes of `gene_names`, `mean`, and `var` should be the same")

    mean[mean == 0] = 1e-12  # set entries equal to zero to small value
    dispersion = var / mean
    # logarithmized mean as in Seurat
    dispersion[dispersion == 0] = np.nan
    dispersion = torch.log(dispersion)

    mean = torch.log1p(mean)
    # All the following quantities are "per-gene" here
    df = pd.DataFrame()
    df["means"] = mean
    df["dispersions"] = dispersion
    df["mean_bin"] = pd.cut(df["means"], bins=n_bins)
    disp_grouped = df.groupby("mean_bin")["dispersions"]
    disp_mean_bin = disp_grouped.mean()
    disp_std_bin = disp_grouped.std(ddof=1)
    df.index = gene_names
    # retrieve those genes that have nan std, these are the ones where
    # only a single gene fell in the bin and implicitly set them to have
    # a normalized disperion of 1
    one_gene_per_bin = disp_std_bin.isnull()
    gen_indices = np.where(one_gene_per_bin[df["mean_bin"].values])[0].tolist()
    if len(gen_indices) > 0:
        logging.debug(
            f"Gene indices {gen_indices} fell into a single bin: their "
            "normalized dispersion was set to 1.\n    "
            "Decreasing `n_bins` will likely avoid this effect."
        )
    # Circumvent pandas 0.23 bug. Both sides of the assignment have dtype==float32,
    # but there’s still a dtype error without “.value”.
    disp_std_bin[one_gene_per_bin.values] = disp_mean_bin[one_gene_per_bin.values].values
    disp_mean_bin[one_gene_per_bin.values] = 0
    # actually do the normalization
    df["dispersions_norm"] = (
        df["dispersions"].values - disp_mean_bin[df["mean_bin"].values].values  # use values here as index differs
    ) / disp_std_bin[df["mean_bin"].values].values

    dispersion_norm = df["dispersions_norm"].values
    if n_top_genes is not None:
        dispersion_norm = dispersion_norm[~np.isnan(dispersion_norm)]
        dispersion_norm[::-1].sort()  # interestingly, np.argpartition is slightly slower
        if n_top_genes > len(gene_names):
            logging.info("`n_top_genes` > `adata.n_var`, returning all genes.")
            n_top_genes = len(gene_names)
        if n_top_genes > len(dispersion_norm):
            warnings.warn(
                "`n_top_genes` > number of normalized dispersions, returning all genes with normalized dispersions.",
                UserWarning,
            )
            n_top_genes = len(dispersion_norm)
        disp_cut_off = dispersion_norm[n_top_genes - 1]
        gene_subset = np.nan_to_num(df["dispersions_norm"].values) >= disp_cut_off
        logging.debug(f"the {n_top_genes} top genes correspond to a " f"normalized dispersion cutoff of {disp_cut_off}")
    else:
        dispersion_norm[np.isnan(dispersion_norm)] = 0  # similar to Seurat
        gene_subset = np.logical_and.reduce(
            (
                mean.numpy() > min_mean,
                mean.numpy() < max_mean,
                dispersion_norm > min_disp,
                dispersion_norm < max_disp,
            )
        )

    df["highly_variable"] = gene_subset
    return df
