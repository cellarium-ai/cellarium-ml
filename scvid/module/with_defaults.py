# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from scvid.module import OnePassMeanVarStd, ProbabilisticPCAPyroModule
from scvid.transforms import ZScoreLog1pNormalize


class OnePassMeanVarStdWithDefaults(OnePassMeanVarStd):
    """Preset default values for the CLI."""

    def __init__(self, target_count: int = 10_000) -> None:
        transform = ZScoreLog1pNormalize(
            mean_g=0, std_g=None, perform_scaling=False, target_count=target_count
        )
        super().__init__(transform=transform)


class ProbabilisticPCAWithDefaults(ProbabilisticPCAPyroModule):
    """Preset default values for the CLI."""
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
            ckpt = torch.load(mean_var_std_ckpt_path)
            onepass = ckpt["hyper_parameters"]["module"]
            # compute W_init_scale and sigma_init_scale
            W_init_scale = torch.sqrt(
                W_init_variance_ratio * onepass.var.sum() / (g_genes * k_components)
            ).item()
            sigma_init_scale = torch.sqrt(
                sigma_init_variance_ratio * onepass.var.sum() / g_genes
            ).item()
            mean_g = onepass.mean
            # create transform
            transform = ZScoreLog1pNormalize(
                mean_g=0, std_g=None, perform_scaling=False, target_count=target_count
            )
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
