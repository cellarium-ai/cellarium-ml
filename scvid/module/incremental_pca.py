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

    Args:
        g_genes: Number of genes.
        k_components: Number of principal components.
        ppca_flavor: Type of the PPCA model. Has to be one of `marginalized` or `linear_vae`.
        mean_g: Mean gene expression of the input data.
        transform: If not ``None`` is used to transform the input data.
    """

    def __init__(
        self,
        g_genes: int,
        k_components: int,
        transform: nn.Module | None = None,
        mean_correct: bool = False,
    ) -> None:
        super().__init__()
        self.g_genes = g_genes
        self.k_components = k_components
        self.transform = transform
        self.mean_correct = mean_correct
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
        k = self.k_components
        m = self.x_size
        n = x_ng.size(0)
        q = min(k + 6, self.g_genes, 2*k)
        #  assert q >= k
        #  assert q <= self.g_genes
        #  assert q >= n

        # compute SVD of new data
        if self.mean_correct:
            x_mean_g = x_ng.mean(dim=0)
            _, S_k, V_kg = torch.linalg.svd(x_ng - x_mean_g)
            # _, S_k, V_kg = torch.svd_lowrank(x_ng - x_mean_g, q=q)
        else:
            _, S_k, V_kg = torch.linalg.svd(x_ng)
            # _, S_k, V_kg = torch.svd_lowrank(x_ng, q=q)

        # if not the first batch, merge results
        if m > 0:
            SV_kg = torch.diag(S_k[:k]) @ V_kg[:k]
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
            _, S_k, V_kg = torch.linalg.svd(C)
            # _, S_k, V_kg = torch.svd_lowrank(C, q=q)
        # update buffers
        self.V_kg = V_kg[:k]
        self.S_k = S_k[:k]
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
