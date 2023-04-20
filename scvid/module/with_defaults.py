# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from scvid.module import OnePassMeanVarStd, ProbabilisticPCAPyroModule
from scvid.transforms import ZScoreLog1pNormalize


class OnePassMeanVarStdWithDefaults(OnePassMeanVarStd):
    """
    Preset default values for the LightningCLI.

    Args:
        target_count: Target gene epxression count. Default: ``10_000
    """

    def __init__(self, target_count: int = 10_000) -> None:
        transform = ZScoreLog1pNormalize(
            mean_g=0, std_g=None, perform_scaling=False, target_count=target_count
        )
        super().__init__(transform=transform)


class ProbabilisticPCAWithDefaults(ProbabilisticPCAPyroModule):
    """
    Preset default values for the LightningCLI.

    Args:
        n_cells: Number of cells.
        g_genes: Number of genes.
        k_components: Number of principcal components.
        ppca_flavor: Type of the PPCA model. Has to be one of `marginalized` or `linear_vae`.
        W_init_variance_ratio: Ratio of variance of W_init_scale to variance of data.
            If ``mean_var_std_ckpt_path`` is ``None``, then ``W_init_scale`` is set to
            ``W_init_variance_ratio``.
        sigma_init_variance_ratio: Ratio of variance of sigma_init_scale to variance of data.
            If ``mean_var_std_ckpt_path`` is ``None``, then ``sigma_init_scale`` is set to
            ``sigma_init_variance_ratio``.
        seed: Random seed used to initialize parameters. Default: ``0``.
        target_count: Target gene epxression count. Default: ``10_000``
        mean_var_std_ckpt_path: Path to checkpoint containing OnePassMeanVarStd.
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
    ):
        if mean_var_std_ckpt_path is None:
            # compute W_init_scale and sigma_init_scale
            W_init_scale = W_init_variance_ratio
            sigma_init_scale = sigma_init_variance_ratio
            mean_g = None
            # create transform
            transform = ZScoreLog1pNormalize(
                mean_g=0, std_g=None, perform_scaling=False, target_count=target_count
            )
        else:
            # load OnePassMeanVarStd from checkpoint
            onepass = torch.load(mean_var_std_ckpt_path)
            # compute W_init_scale and sigma_init_scale
            W_init_scale = torch.sqrt(
                W_init_variance_ratio * onepass.var_g.sum() / (g_genes * k_components)
            ).item()
            sigma_init_scale = torch.sqrt(
                sigma_init_variance_ratio * onepass.var_g.sum() / g_genes
            ).item()
            mean_g = onepass.mean_g
            # create transform
            transform = ZScoreLog1pNormalize(
                mean_g=0, std_g=None, perform_scaling=False, target_count=target_count
            )
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
            transform=transform,
        )
