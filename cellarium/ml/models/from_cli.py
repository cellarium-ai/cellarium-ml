# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math

import torch

from cellarium.ml.models.onepass_mean_var_std import OnePassMeanVarStd
from cellarium.ml.models.probabilistic_pca import ProbabilisticPCA
from cellarium.ml.transforms import NormalizeTotal


class ProbabilisticPCAFromCLI(ProbabilisticPCA):
    """
    Preset default values for the LightningCLI.

    Args:
        n_cells:
            Number of cells.
        g_genes:
            Number of genes.
        k_components:
            Number of principcal components.
        ppca_flavor:
            Type of the PPCA model. Has to be one of ``marginalized`` or ``linear_vae``.
        W_init_variance_ratio:
            Ratio of variance of W_init_scale to variance of data.
            If ``mean_var_std_ckpt_path`` is ``None``, then ``W_init_scale`` is set to
            ``W_init_variance_ratio``.
        sigma_init_variance_ratio:
            Ratio of variance of sigma_init_scale to variance of data.
            If ``mean_var_std_ckpt_path`` is ``None``, then ``sigma_init_scale`` is set to
            ``sigma_init_variance_ratio``.
        seed:
            Random seed used to initialize parameters.
        target_count:
            Target gene expression count.
        mean_var_std_ckpt_path:
            Path to checkpoint containing OnePassMeanVarStd.
    """

    def __init__(
        self,
        n_cells: int,
        g_genes: int,
        k_components: int = 256,
        ppca_flavor: str = "marginalized",
        W_init_variance_ratio: float = 0.5,
        sigma_init_variance_ratio: float = 0.5,
        seed: int = 0,
        target_count: int = 10_000,
        mean_var_std_ckpt_path: str | None = None,
    ) -> None:
        if mean_var_std_ckpt_path is None:
            # compute W_init_scale and sigma_init_scale
            W_init_scale = W_init_variance_ratio
            sigma_init_scale = sigma_init_variance_ratio
            mean_g = None
        else:
            # load OnePassMeanVarStdFromCLI from checkpoint
            onepass: OnePassMeanVarStd = torch.load(mean_var_std_ckpt_path)
            assert isinstance(onepass.transform, torch.nn.Sequential)
            assert isinstance(onepass.transform[0], NormalizeTotal)
            assert target_count == onepass.transform.target_count
            assert g_genes == onepass.g_genes
            # compute W_init_scale and sigma_init_scale
            total_variance = onepass.var_g.sum().item()
            W_init_scale = math.sqrt(W_init_variance_ratio * total_variance / (g_genes * k_components))
            sigma_init_scale = math.sqrt(sigma_init_variance_ratio * total_variance / g_genes)
            mean_g = onepass.mean_g
        # create transform
        self.mean_var_std_ckpt_path = mean_var_std_ckpt_path
        super().__init__(
            n_cells=n_cells,
            g_genes=g_genes,
            k_components=k_components,
            ppca_flavor=ppca_flavor,
            mean_g=mean_g,
            W_init_scale=W_init_scale,
            sigma_init_scale=sigma_init_scale,
            seed=seed,
        )
