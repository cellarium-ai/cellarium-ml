# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math

import torch
import torch.nn as nn

from scvid.module import BaseModule


class IncrementalPCA(BaseModule):
    """
    Incremental PCA.

    **Reference:**

    1. *A Distributed and Incremental SVD Algorithm for Agglomerative Data Analysis on Large Networks*,
       M. A. Iwen, B. W. Ong
       (https://users.math.msu.edu/users/iwenmark/Papers/distrib_inc_svd.pdf)
    2. *Incremental Learning for Robust Visual Tracking*,
       D. Ross, J. Lim, R.-S. Lin, M.-H. Yang
       (https://www.cs.toronto.edu/~dross/ivt/RossLimLinYang_ijcv.pdf)

    Args:
        g_genes: Number of genes.
        k_components: Number of principal components.
        p_oversamples: Additional number of random vectors to sample the range of ``x_ng``
            so as to ensure proper conditioning.
        transform: If not ``None`` is used to transform the input data.
        mean_correct: If ``True`` then the mean correction is applied to the update step.
    """

    def __init__(
        self,
        g_genes: int,
        k_components: int,
        p_oversamples: int = 10,
        mean_correct: bool = False,
        transform: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.g_genes = g_genes
        self.k_components = k_components
        self.p_oversamples = p_oversamples
        self.mean_correct = mean_correct
        self.transform = transform
        self.V_kg: torch.Tensor
        self.S_k: torch.Tensor
        self.x_mean_g: torch.Tensor
        self.x_size: torch.Tensor
        self.register_buffer("V_kg", torch.zeros(k_components, g_genes))
        self.register_buffer("S_k", torch.zeros(k_components))
        self.register_buffer("x_mean_g", torch.zeros(g_genes))
        self.register_buffer("x_size", torch.tensor(0))
        self._dummy_param = torch.nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _get_fn_args_from_batch(
        tensor_dict: dict[str, torch.Tensor]
    ) -> tuple[tuple, dict]:
        x = tensor_dict["X"]
        return (x,), {}

    def forward(self, x_ng: torch.Tensor) -> None:
        if self.transform is not None:
            x_ng = self.transform(x_ng)

        g = self.g_genes
        k = self.k_components
        p = self.p_oversamples
        m = self.x_size
        n = x_ng.size(0)
        assert p + k <= min(n, g)

        # compute SVD of new data
        if self.mean_correct:
            x_mean_g = x_ng.mean(dim=0)
            _, S_q, V_gq = torch.svd_lowrank(x_ng - x_mean_g, q=k + p)
        else:
            _, S_q, V_gq = torch.svd_lowrank(x_ng, q=k + p)

        # if not the first batch, merge results
        if m > 0:
            SV_kg = torch.diag(S_q[:k]) @ V_gq.T[:k]
            if self.mean_correct:
                mean_correction = (
                    math.sqrt(m * n / (m + n)) * (self.x_mean_g - x_mean_g)[None, :]
                )
                C = torch.cat(
                    [torch.diag(self.S_k) @ self.V_kg, SV_kg, mean_correction],
                    dim=0,
                )
            else:
                C = torch.cat([torch.diag(self.S_k) @ self.V_kg, SV_kg], dim=0)
            # perform SVD on merged results
            _, S_q, V_gq = torch.svd_lowrank(C, q=k + p)

        # update buffers
        self.V_kg = V_gq.T[:k]
        self.S_k = S_q[:k]
        self.x_size = m + n
        if self.mean_correct:
            self.x_mean_g = self.x_mean_g * m / (m + n) + x_mean_g * n / (m + n)

    @property
    def L_k(self) -> torch.Tensor:
        r"""
        Vector with elements given by the PC eigenvalues.
        """
        return self.S_k**2 / self.x_size

    @property
    def U_gk(self) -> torch.Tensor:
        r"""
        Principal components corresponding to eigenvalues ``L_k``.
        """
        return self.V_kg.T
